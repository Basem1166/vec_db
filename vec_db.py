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
    # 4. RETRIEVAL
    # -------------------------------------------------------------------------
    def retrieve(self, query: np.ndarray, top_k=5):
        query = query.reshape(1, -1).astype(np.float32)
        q_norm = np.linalg.norm(query)

        VEC_BYTES = DIMENSION * 4

        num_records = self._get_num_records()
        n_probes = 5 if num_records <= 1_000_000 else 10

        # --- A. Read index metadata ---
        with open(self.index_path, "rb") as f:
            n_clusters = struct.unpack("I", f.read(4))[0]

            centroid_bytes = f.read(n_clusters * DIMENSION * 4)
            centroids = np.frombuffer(centroid_bytes, dtype=np.float32) \
                        .reshape(n_clusters, DIMENSION)

            table_bytes = f.read(n_clusters * 8)
            cluster_table = np.frombuffer(table_bytes, dtype=np.int32) \
                            .reshape(n_clusters, 2)

            # --- B. Coarse search ---
            c_norms = np.linalg.norm(centroids, axis=1)
            dots = np.dot(centroids, query.T).flatten()
            sims = dots / (c_norms * q_norm + 1e-9)
            closest_clusters = np.argsort(sims)[::-1][:n_probes]

        # --- C. Fine search (true streaming, no memmap indexing) ---
        best_scores = []
        best_ids = []

        with open(self.db_path, "rb", buffering=0) as dbf:   # unbuffered (fast seeks)
            for cid in closest_clusters:
                offset, count = cluster_table[cid]
                if count == 0:
                    continue

                with open(self.index_path, "rb") as fidx:
                    fidx.seek(offset)
                    ids_bytes = fidx.read(count * 4)
                    row_ids = np.frombuffer(ids_bytes, dtype=np.int32, count=count)

                # --- Stream vectors one by one ---
                for rid in row_ids:
                    # seek to vector position
                    rid = int(rid)
                    pos = rid * VEC_BYTES
                    os.lseek(dbf.fileno(), pos, os.SEEK_SET)


                    # read 64 floats (256 bytes)
                    raw = dbf.read(VEC_BYTES)
                    vec = np.frombuffer(raw, dtype=np.float32)

                    # cosine similarity
                    v_norm = np.linalg.norm(vec)
                    score = np.dot(vec, query.T).item() / (v_norm * q_norm + 1e-9)

                    # keep top-k list very small
                    if len(best_scores) < top_k:
                        best_scores.append(score)
                        best_ids.append(rid)
                    else:
                        min_idx = np.argmin(best_scores)
                        if score > best_scores[min_idx]:
                            best_scores[min_idx] = score
                            best_ids[min_idx] = rid

        # --- D. Final sorting ---
        order = np.argsort(best_scores)[::-1]
        return [best_ids[i] for i in order]







