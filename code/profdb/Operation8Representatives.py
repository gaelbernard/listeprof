import os, logging, duckdb
from code.profdb.Operation0Abstract import OperationAbstract
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import KMeans

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
from pathlib import Path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(Path(PROJECT_ROOT) / ".env")

class OperationRepresentative(OperationAbstract):
    def __init__(self, db_path: str):
        super().__init__(db_path)

    def pre(self):

        # Create the representative table
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS representative (
                sciper INTEGER,
                type VARCHAR,           -- 'mean', 'recency', 'cluster'
                weight FLOAT,           -- 1.0 for mean/recency, cluster_size/total for clusters
                embedding FLOAT[]
            )
        """)
        # Clear existing data for fresh run
        self.con.execute("DELETE FROM representative")

    def _compute_mean(self, embeddings: np.ndarray) -> tuple[np.ndarray, float]:
        """Simple mean embedding with weight=1"""
        return np.mean(embeddings, axis=0), 1.0

    def _compute_recency_weighted(self, embeddings: np.ndarray, years: np.ndarray, half_life: float = 3.0) -> tuple[
        np.ndarray, float]:
        """
        Recency-weighted mean using exponential decay.
        half_life: number of years for weight to halve (default 3 years)
        """
        current_year = pd.Timestamp.now().year
        ages = current_year - years
        # Exponential decay: weight = 2^(-age/half_life)
        weights = np.power(2, -ages / half_life)
        # Normalize weights
        weights = weights / weights.sum()
        # Weighted average
        weighted_embedding = np.average(embeddings, axis=0, weights=weights)
        return weighted_embedding, 1.0

    def _compute_clusters(self, embeddings: np.ndarray, max_clusters: int = 5) -> list[tuple[np.ndarray, float]]:
        """
        Cluster embeddings and return centroid for each cluster.
        Uses min(n_samples, max_clusters) clusters.
        Returns list of (embedding, weight) tuples.
        """
        n_samples = len(embeddings)
        n_clusters = min(n_samples, max_clusters)

        if n_clusters == 1:
            # Single publication: just return it
            return [(embeddings[0], 1.0)]

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        results = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_size = mask.sum()
            if cluster_size > 0:
                centroid = np.mean(embeddings[mask], axis=0)
                weight = cluster_size / n_samples
                results.append((centroid, weight))

        return results

    def trans(self):
        scipers = self.con.execute("SELECT DISTINCT sciper FROM prof").df()['sciper']
        total_representatives = 0

        for sciper in scipers:
            # Retrieve embeddings and years for this professor
            data = self.con.execute(f"""
                SELECT pub.year_issued, embedding 
                FROM pub_embedding 
                INNER JOIN sciper_pub USING (id_pub) 
                INNER JOIN pub USING (id_pub) 
                WHERE sciper = {sciper}
            """).df()

            if len(data) == 0:
                continue

            embeddings = np.array(data['embedding'].tolist())
            years = data['year_issued'].values

            # 1. Mean embedding
            mean_emb, mean_weight = self._compute_mean(embeddings)
            self.con.execute("""
                INSERT INTO representative (sciper, type, weight, embedding)
                VALUES (?, 'mean', ?, ?)
            """, [int(sciper), mean_weight, mean_emb.tolist()])
            total_representatives += 1

            # 2. Recency-weighted mean
            recency_emb, recency_weight = self._compute_recency_weighted(embeddings, years)
            self.con.execute("""
                INSERT INTO representative (sciper, type, weight, embedding)
                VALUES (?, 'recency', ?, ?)
            """, [int(sciper), recency_weight, recency_emb.tolist()])
            total_representatives += 1

            # 3. Cluster-based embeddings (adaptive k)
            cluster_results = self._compute_clusters(embeddings)
            for cluster_emb, cluster_weight in cluster_results:
                self.con.execute("""
                    INSERT INTO representative (sciper, type, weight, embedding)
                    VALUES (?, 'cluster', ?, ?)
                """, [int(sciper), cluster_weight, cluster_emb.tolist()])
                total_representatives += 1

        logging.info(f"Total representative embeddings created: {total_representatives}")


if __name__ == "__main__":
    DB = "/data/gael/2025-10-08-listProf/output/latest/db.duckdb"
    OperationRepresentative(DB).run()