import numpy as np
import os
import struct
from sklearn.cluster import MiniBatchKMeans
from typing import Annotated

# Constants
DIMENSION = 70
ELEMENT_SIZE = 4  # float32 is 4 bytes
DB_SEED_NUMBER = 42

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", 
                 centroids_file_path="centroids.npy", offsets_file_path="offsets.npy",
                 new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.centroids_path = centroids_file_path
        self.offsets_path = offsets_file_path
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # Clean up old files
            for p in [self.db_path, self.index_path, self.centroids_path, self.offsets_path]:
                if os.path.exists(p):
                    os.remove(p)
            self.generate_database(db_size)

    def get_one_row(self, row_num: int) -> np.ndarray:
      try:
          offset = row_num * DIMENSION * ELEMENT_SIZE
          # Using memmap to read just 1 small slice from disk
          mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                                  shape=(1, DIMENSION), offset=offset)
          return np.array(mmap_vector[0])
      except Exception as e:
          print(f"Error getting row {row_num}: {e}")
          return np.zeros(DIMENSION)

    def get_all_rows(self) -> np.ndarray:
      """
      Loads the entire database into memory.
      WARNING: High RAM usage (approx 5.6GB for 20M vectors).
      Used by evaluation scripts for Ground Truth calculation.
      """
      
      num_records = self._get_num_records()
      vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
      print("[EVAL] Got All Rows.")
      return np.array(vectors)

    def generate_database(self, size: int) -> None:
        """
        Generates random vectors and writes them to disk in chunks to save RAM.
        """
        print(f"[DB] Generating {size} vectors...")
        rng = np.random.default_rng(DB_SEED_NUMBER)
        
        # 1. Create the file on disk first
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=(size, DIMENSION))
        
        # 2. Write in chunks (prevents Memory Error)
        chunk_size = 500_000
        for i in range(0, size, chunk_size):
            end = min(i + chunk_size, size)
            # Generate only what fits in RAM
            mmap_vectors[i:end] = rng.random((end - i, DIMENSION), dtype=np.float32)
            if i % 1_000_000 == 0:
                mmap_vectors.flush() # Periodic save
        
        mmap_vectors.flush()
        print("[DB] Generation complete.")
        self._build_index()

    def _get_num_records(self) -> int:
        if not os.path.exists(self.db_path): return 0
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: np.ndarray):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        
        # Expand the memmap
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        
        # Rebuild index (Simple approach: full rebuild. 
        # Incremental updates are complex without specialized libs)
        self._build_index()

    def _build_index(self):
        num_records = self._get_num_records()
        print(f"[INDEX] Building IVF index for {num_records} vectors...")

        # 1. Determine n_clusters dynamically (Rule of thumb: sqrt(N))
        # More data = More clusters needed to keep search fast
        if num_records <= 1_000_000:
            n_clusters = 1000
        elif num_records <= 10_000_000:
            n_clusters = 3000  # Approx sqrt(1M)
        elif num_records <= 15_000_000:
            n_clusters = 4000
        else:
            n_clusters =5000
            

        # 2. Train K-Means (Pass 1)
        # MiniBatchKMeans is much faster and memory efficient
        print("[INDEX] Training K-Means...")
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            batch_size=10000, 
            random_state=DB_SEED_NUMBER,
            n_init='auto'
        )
        
        # We assume dataset fits in virtual memory (mmap handles this). 
        # If dataset is HUGE (>RAM), MiniBatchKMeans handles it via partial_fit, 
        # but fit(mmap) works fine for <100GB usually.
        train_size = min(500_000, num_records)
        kmeans.fit(mmap_vectors[:train_size])
        
        np.save(self.centroids_path, kmeans.cluster_centers_)
        print("[INDEX] Centroids saved.")

        # 3. Assign Vectors to Clusters (Pass 2)
        # CRITICAL OPTIMIZATION: Do not open N files. 
        # Predict all labels, sort them, then write sequentially.
        print("[INDEX] Assigning vectors to clusters...")
        
        # Predict in chunks to be safe
        batch_size = 50000
        all_labels = np.zeros(num_records, dtype=np.int32)
        
        for i in range(0, num_records, batch_size):
            end = min(i + batch_size, num_records)
            batch = mmap_vectors[i:end]
            all_labels[i:end] = kmeans.predict(batch)

        # 4. Sort to group by Cluster ID
        # argsort gives us the indices that would sort the array
        print("[INDEX] Sorting and writing inverted lists...")
        sorted_indices = np.argsort(all_labels) # This groups all cluster 0, then 1, etc.
        sorted_labels = all_labels[sorted_indices]

        # 5. Write to Index File & Create Offset Map
        offsets = {} # Store where each cluster starts in the file
        
        with open(self.index_path, "wb") as f:
            # Iterate through each cluster
            # We use searchsorted to find boundaries in the sorted array efficiently
            for cid in range(n_clusters):
                # Find start and end index of this cluster in the sorted array
                start_idx = np.searchsorted(sorted_labels, cid, side='left')
                end_idx = np.searchsorted(sorted_labels, cid, side='right')
                
                # These are the actual Row IDs belonging to cluster `cid`
                ids_in_cluster = sorted_indices[start_idx:end_idx].astype(np.int32)
                
                # Record the file offset (bytes) where this list starts
                offsets[cid] = f.tell()
                
                # Write Size (4 bytes) + IDs (N * 4 bytes)
                f.write(struct.pack("I", len(ids_in_cluster)))
                if len(ids_in_cluster) > 0:
                    ids_in_cluster.tofile(f)

        # Save offsets so we can jump instantly during retrieve
        np.save(self.offsets_path, offsets)
        print(f"[INDEX] Index built with {n_clusters} clusters.")

    def retrieve(self, query: np.ndarray, top_k=5):
        # Ensure query is (1, DIMENSION)
        query = query.reshape(1, -1)
        
        # Load metadata
        centroids = np.load(self.centroids_path)
        offsets = np.load(self.offsets_path, allow_pickle=True).item()
        
        num_records = self._get_num_records()
        
        # Determine n_probe based on DB size
        if num_records <= 1_000_000: n_probes = 5
        else: n_probes = 10  # Search more buckets for larger data

        # ---- 1. Coarse Quantization (Find closest centroids) ----
        # Calculate Cosine Similarity to centroids
        # Sim = (A . B) / (|A| * |B|)
        # Pre-calculating norms speeds this up
        c_norms = np.linalg.norm(centroids, axis=1)
        q_norm = np.linalg.norm(query)
        
        # Dot product
        dists = np.dot(centroids, query.T).flatten()
        # Divide by norms (Cosine Sim)
        sims = dists / (c_norms * q_norm + 1e-10) # Avoid div by zero
        
        # Get top n_probes clusters (highest similarity)
        closest_clusters = np.argsort(sims)[::-1][:n_probes]

        # ---- 2. Fine Search (Scan candidates) ----
        candidates_scores = []
        candidates_ids = []

        # Open Main DB for reading vectors (Keep open for loop)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        
        with open(self.index_path, "rb") as f:
            for cid in closest_clusters:
                if cid not in offsets: continue
                
                # JUMP directly to the cluster list
                f.seek(offsets[cid])
                
                # Read size
                size_bytes = f.read(4)
                if not size_bytes: continue
                size = struct.unpack("I", size_bytes)[0]
                
                if size == 0: continue
                
                # Read all IDs in this cluster at once
                ids_binary = f.read(size * 4)
                row_ids = np.frombuffer(ids_binary, dtype=np.int32)
                
                # ---- VECTORIZED SCORING (The huge speedup) ----
                # Instead of loop, load all vectors for this cluster
                cluster_vecs = mmap_vectors[row_ids]
                
                # Calculate scores for all vectors in this cluster at once
                # shape: (N_cluster, )
                c_vec_norms = np.linalg.norm(cluster_vecs, axis=1)
                dot_products = np.dot(cluster_vecs, query.T).flatten()
                batch_scores = dot_products / (c_vec_norms * q_norm + 1e-10)
                
                candidates_scores.extend(batch_scores)
                candidates_ids.extend(row_ids)

        # ---- 3. Final Top-K ----
        candidates_scores = np.array(candidates_scores)
        candidates_ids = np.array(candidates_ids)
        
        if len(candidates_scores) == 0:
            return []
            
        # Get top K indices
        # argpartition is faster than sort for finding top K
        if len(candidates_scores) > top_k:
            top_indices = np.argpartition(candidates_scores, -top_k)[-top_k:]
            # Sort the top K explicitly (argpartition doesn't guarantee order within top K)
            sorted_top_indices = top_indices[np.argsort(candidates_scores[top_indices])[::-1]]
        else:
            sorted_top_indices = np.argsort(candidates_scores)[::-1]
            
        return candidates_ids[sorted_top_indices].tolist()
