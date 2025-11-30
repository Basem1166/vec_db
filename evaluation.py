from typing import List
from dataclasses import dataclass
import numpy as np
def run_queries_safe(db, top_k, num_runs):
    """
    Calculates Ground Truth in small batches.
    UPDATED: Uses 64 dimensions to match your new database.
    """
    results = []
    
    # 1. Get correct dimensions dynamically from the DB file size
    # or just use the constant 64.
    dim = 64 
    
    num_records = db._get_num_records()
    
    # FIX: Changed shape from 70 to 64
    mmap_vectors = np.memmap(db.db_path, dtype=np.float32, mode='r', shape=(num_records, dim))
    
    print(f"[SAFE EVAL] Running {num_runs} queries on {num_records} vectors (Dim={dim})...")

    for i in range(num_runs):
        # FIX: Generate query with correct dimension (64)
        query = np.random.random((1, dim)).astype(np.float32)

        # A. Measure Your Index Speed
        tic = time.time()
        db_ids = db.retrieve(query, top_k)
        toc = time.time()
        run_time = toc - tic

        # B. Calculate Ground Truth (Batched)
        batch_size = 500_000
        candidates = []
        q_norm = np.linalg.norm(query)

        for start in range(0, num_records, batch_size):
            end = min(start + batch_size, num_records)
            batch = mmap_vectors[start:end]
            
            norms = np.linalg.norm(batch, axis=1)
            dots = np.dot(batch, query.T).flatten()
            scores = dots / (norms * q_norm + 1e-10)
            
            top_batch_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_batch_indices:
                candidates.append((scores[idx], start + idx))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        actual_ids = [c[1] for c in candidates[:top_k]]

        results.append(Result(run_time, top_k, db_ids, actual_ids))
        print(f"   -> Query {i+1} finished.")

    return results

# --- 1. Rename 'eval' to avoid conflict with Python's built-in ---
def evaluate_metrics(results: List[Result]):
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        # Case for retrieving number not equal to top_k, score will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)

# --- 2. Run the Evaluation ---
if __name__ == "__main__":
    # Ensure new_db=False so we load the one you already built
    db = VecDB(db_size=10**6, new_db=False,database_file_path="OpenSubtitles_en_1M_emb_64.dat")
    VecDB._build_index(db)
    
    # Run the queries (assuming you have run_queries_safe defined from the previous step)
    res = run_queries_safe(db, top_k=5, num_runs=5)
    
    # CALL THE RENAMED FUNCTION
    score, avg_time = evaluate_metrics(res)
    
    print(f"\nFinal Score: {score}")
    print(f"Avg Time: {avg_time}")
