import numpy as np
import os
import struct
from sklearn.cluster import MiniBatchKMeans
from typing import Annotated

# Constants
DIMENSION = 64
ELEMENT_SIZE = 4  # float32 is 4 bytes
DB_SEED_NUMBER = 42

class VecDB:
    # -------------------------------------------------------------------------
    # 1. STRICT INIT SIGNATURE (Do not change this)
    # -------------------------------------------------------------------------
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", 
                 new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            
            # Clean up old files
            if os.path.exists(self.db_path): os.remove(self.db_path)
            if os.path.exists(self.index_path): os.remove(self.index_path)
            
            self.generate_database(db_size)

    # -------------------------------------------------------------------------
    # 2. FILE OPERATIONS
    # -------------------------------------------------------------------------
    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                                    shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return np.zeros(DIMENSION)

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def _get_num_records(self) -> int:
        if not os.path.exists(self.db_path): return 0
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    # -------------------------------------------------------------------------
    # 3. GENERATION & INDEXING
    # -------------------------------------------------------------------------
    def generate_database(self, size: int) -> None:
        print(f"[DB] Generating {size} vectors...")
        rng = np.random.default_rng(DB_SEED_NUMBER)
        
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=(size, DIMENSION))
        
        chunk_size = 500_000
        for i in range(0, size, chunk_size):
            end = min(i + chunk_size, size)
            mmap_vectors[i:end] = rng.random((end - i, DIMENSION), dtype=np.float32)
            if i % 1_000_000 == 0: mmap_vectors.flush()
        
        mmap_vectors.flush()
        print("[DB] Generation complete.")
        self._build_index()

    def _build_index(self):
        num_records = self._get_num_records()
        print(f"[INDEX] Building Single-File Index for {num_records} vectors...")

        # A. Determine clusters
        if num_records <= 1_000_000: n_clusters = 1000
        elif num_records <= 10_000_000: n_clusters = 3000
        else: n_clusters = 5000

        # B. Train K-Means (Subsampling)
        print("[INDEX] Training K-Means...")
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, 
                                 random_state=DB_SEED_NUMBER, n_init='auto')
        
        train_size = min(500_000, num_records)
        kmeans.fit(mmap_vectors[:train_size])
        
        centroids = kmeans.cluster_centers_.astype(np.float32)

        # C. Assign Vectors
        print("[INDEX] Assigning vectors...")
        batch_size = 100000
        all_labels = np.zeros(num_records, dtype=np.int32)
        
        for i in range(0, num_records, batch_size):
            end = min(i + batch_size, num_records)
            all_labels[i:end] = kmeans.predict(mmap_vectors[i:end])

        # D. Sort IDs
        print("[INDEX] Sorting lists...")
        sorted_indices = np.argsort(all_labels)
        sorted_labels = all_labels[sorted_indices]

        # E. WRITE SINGLE INDEX FILE
        # Format: 
        # [N_Clusters (int)] 
        # [Centroids (N*Dim floats)] 
        # [Offset_Table (N*2 ints -> start_byte, count)]
        # [Inverted Lists (Integers...)]
        
        print(f"[INDEX] Writing to {self.index_path}...")
        with open(self.index_path, "wb") as f:
            # 1. Write Header: Number of Clusters
            f.write(struct.pack("I", n_clusters))
            
            # 2. Write Centroids
            f.write(centroids.tobytes())
            
            # 3. Reserve space for Offset Table 
            # Each entry is 2 ints (start_offset, count) -> 8 bytes
            table_offset_start = f.tell()
            f.write(b'\0' * (n_clusters * 8))
            
            # 4. Write Inverted Lists & Record Offsets
            cluster_metadata = [] # Stores (offset, count)
            
            for cid in range(n_clusters):
                # Find range in sorted array
                start_idx = np.searchsorted(sorted_labels, cid, side='left')
                end_idx = np.searchsorted(sorted_labels, cid, side='right')
                
                count = end_idx - start_idx
                current_file_pos = f.tell()
                
                # Store metadata (Where this list starts, How many items)
                cluster_metadata.append((current_file_pos, count))
                
                if count > 0:
                    ids = sorted_indices[start_idx:end_idx].astype(np.int32)
                    f.write(ids.tobytes())

            # 5. Go back and fill in the Offset Table
            f.seek(table_offset_start)
            for offset, count in cluster_metadata:
                f.write(struct.pack("II", offset, count))

        print("[INDEX] Done.")

    # -------------------------------------------------------------------------
    # 4. RETRIEVAL (Memory-optimized: <50MB RAM)
    # -------------------------------------------------------------------------
    def retrieve(self, query: np.ndarray, top_k=5):
        query = query.reshape(-1).astype(np.float32)
        q_norm = np.linalg.norm(query)

        num_records = self._get_num_records()
        if num_records <= 1_000_000:
            n_probes = 5
        else:
            n_probes = 10

        # Smaller batch size to reduce peak memory (~256KB per batch for vectors)
        BATCH = 1000
        # Batch size for reading IDs to avoid loading huge ID arrays
        ID_BATCH = 5000

        with open(self.index_path, "rb") as f:
            # --- A. Read header ---
            n_clusters = struct.unpack("I", f.read(4))[0]
            centroids_start = f.tell()
            table_start = centroids_start + n_clusters * DIMENSION * 4

            # --- B. Coarse search: stream centroids to find closest clusters ---
            # Process centroids in chunks to avoid loading all at once
            CENTROID_BATCH = 500
            cluster_sims = np.empty(n_clusters, dtype=np.float32)
            
            for c_start in range(0, n_clusters, CENTROID_BATCH):
                c_end = min(c_start + CENTROID_BATCH, n_clusters)
                c_count = c_end - c_start
                
                f.seek(centroids_start + c_start * DIMENSION * 4)
                chunk_bytes = f.read(c_count * DIMENSION * 4)
                chunk = np.frombuffer(chunk_bytes, dtype=np.float32).reshape(c_count, DIMENSION)
                
                c_norms = np.linalg.norm(chunk, axis=1)
                dots = np.dot(chunk, query)
                cluster_sims[c_start:c_end] = dots / (c_norms * q_norm + 1e-9)
                
                del chunk, c_norms, dots
            
            closest_clusters = np.argpartition(cluster_sims, -n_probes)[-n_probes:]
            closest_clusters = closest_clusters[np.argsort(cluster_sims[closest_clusters])[::-1]]
            del cluster_sims

            # --- C. Read only needed entries from offset table ---
            cluster_info = []
            for cid in closest_clusters:
                f.seek(table_start + cid * 8)
                offset, count = struct.unpack("II", f.read(8))
                if count > 0:
                    cluster_info.append((cid, offset, count))

            # --- D. Fine search with streaming ---
            # Use fixed-size numpy arrays instead of lists for efficiency
            max_keep = max(10 * top_k, 100)  # Keep a small buffer
            best_scores = np.full(max_keep, -np.inf, dtype=np.float32)
            best_ids = np.zeros(max_keep, dtype=np.int32)
            n_found = 0

            # Memory-map DB (this is just a view, doesn't load into RAM)
            mmap_db = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                               shape=(num_records, DIMENSION))

            for cid, offset, count in cluster_info:
                # Stream IDs in smaller chunks to avoid huge allocations
                for id_start in range(0, count, ID_BATCH):
                    id_end = min(id_start + ID_BATCH, count)
                    id_count = id_end - id_start
                    
                    # Read only this chunk of IDs
                    f.seek(offset + id_start * 4)
                    id_bytes = f.read(id_count * 4)
                    row_ids = np.frombuffer(id_bytes, dtype=np.int32)

                    # Process vectors in small batches
                    for i in range(0, len(row_ids), BATCH):
                        batch_ids = row_ids[i:i+BATCH]
                        
                        # Copy vectors to contiguous array (required for non-contiguous memmap access)
                        vecs = np.empty((len(batch_ids), DIMENSION), dtype=np.float32)
                        for j, rid in enumerate(batch_ids):
                            vecs[j] = mmap_db[rid]

                        # Compute cosine similarity
                        norms = np.linalg.norm(vecs, axis=1)
                        dots = np.dot(vecs, query)
                        sims = dots / (norms * q_norm + 1e-9)
                        
                        del vecs, norms, dots

                        # Update best results
                        for j, sim in enumerate(sims):
                            if n_found < max_keep:
                                best_scores[n_found] = sim
                                best_ids[n_found] = batch_ids[j]
                                n_found += 1
                            elif sim > best_scores.min():
                                min_idx = np.argmin(best_scores)
                                best_scores[min_idx] = sim
                                best_ids[min_idx] = batch_ids[j]
                        
                        del sims

                    del row_ids

            del mmap_db

        # --- E. Final top-k selection ---
        if n_found == 0:
            return []

        valid_scores = best_scores[:n_found]
        valid_ids = best_ids[:n_found]

        if n_found > top_k:
            idx = np.argpartition(valid_scores, -top_k)[-top_k:]
            idx = idx[np.argsort(valid_scores[idx])[::-1]]
        else:
            idx = np.argsort(valid_scores)[::-1]

        return valid_ids[idx].tolist()






