import os, logging
from code.profdb.Operation0Abstract import OperationAbstract
import pandas as pd
import numpy as np
from code.profdb.EmbeddingService import EmbeddingService

from dotenv import load_dotenv
from pathlib import Path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(Path(PROJECT_ROOT) / ".env")


class OperationEmbedding(OperationAbstract):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.embedding_service = None

    def pre(self):
        cache_path = os.path.join(PROJECT_ROOT, "cache_embeddings.lmdb")
        self.embedding_service = EmbeddingService(cache_path=cache_path)

        self.con.execute("""DROP TABLE IF EXISTS pub_embedding""")
        self.con.execute("""
             CREATE TABLE pub_embedding
             (
                 id_pub BIGINT,
                 embedding DOUBLE[]
             )
             """)

    def trans(self):
        BATCH_SIZE_DB = 1000
        min_id, max_id = self.con.execute("""
              SELECT min(id_pub), max(id_pub)
              FROM main.pub
              """).fetchone()
        current = min_id

        while current <= max_id:
            upper = current + BATCH_SIZE_DB - 1
            df = self.con.execute(f"""
                SELECT id_pub, CONCAT(IFNULL(title, ''), '. ', IFNULL(abstract, '')) AS text
                FROM main.pub
                WHERE id_pub BETWEEN {current} AND {upper}
                ORDER BY id_pub
            """).df()

            if len(df) == 0:
                current += BATCH_SIZE_DB
                continue

            # Embed all texts in batch (service handles caching internally)
            embeddings = self.embedding_service.embed(df['text'].tolist())

            # Prepare for insert
            df_emb = pd.DataFrame({
                'id_pub': df['id_pub'],
                'embedding': [emb.tolist() for emb in embeddings]
            })

            self.con.register("embeddings_df", df_emb)
            self.con.execute("""
                 INSERT INTO pub_embedding (id_pub, embedding)
                 SELECT id_pub, embedding
                 FROM embeddings_df
                 """)

            current += BATCH_SIZE_DB

        total = self.con.execute("SELECT COUNT(*) FROM pub_embedding").fetchone()[0]
        logging.info(f"Total embeddings stored: {total}")


if __name__ == "__main__":
    DB = "/data/gael/2025-10-08-listProf/output/db_20251209_041119/db.duckdb"
    OperationEmbedding(DB).run()