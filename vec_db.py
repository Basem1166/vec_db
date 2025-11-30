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
    def retrieve(self, query, top_k):
        # compute query hash code
        query_code = self.compute_hash(query)
    
        # get candidate row IDs from hash buckets
        row_ids = self.hash_table.get(query_code, [])
    
        # if no matches in bucket
        if not row_ids:
            return []
    
        VEC_BYTES = self.vec_dim * 4  # float32
    
        distances = []
    
        # open vector file
        with open(self.vec_file, "rb") as dbf:
            for rid in row_ids:
    
                # ensure Python int to avoid numpy int32 overflow
                rid = int(rid)
    
                # bounds check to prevent invalid lseek()
                if rid < 0 or rid >= self.num_vectors:
                    # This SHOULD NEVER HAPPEN — but if it does,
                    # it means your index stored corrupted row IDs.
                    print(f"[ERROR] Invalid rid={rid} (valid range 0..{self.num_vectors-1})")
                    continue
    
                pos = rid * VEC_BYTES
    
                # extra safety: ensure position is within file
                if pos < 0 or pos + VEC_BYTES > self.vec_file_size:
                    print(f"[ERROR] Invalid file seek position pos={pos}, rid={rid}")
                    continue
    
                # seek to correct vector location
                os.lseek(dbf.fileno(), pos, os.SEEK_SET)
    
                # read vector bytes
                raw = dbf.read(VEC_BYTES)
                vec = np.frombuffer(raw, dtype=np.float32)
    
                # compute L2 distance
                dist = np.dot(query - vec, query - vec)
                distances.append((dist, rid))
    
        # get top K by smallest distance
        distances.sort(key=lambda x: x[0])
        return [rid for _, rid in distances[:top_k]]







