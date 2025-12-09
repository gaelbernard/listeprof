from datetime import datetime
import os
from abc import abstractmethod
import pandas as pd
import logging
from pathlib import Path
from code.profdb.Operation1Prof import OperationProf
from code.profdb.Operation2OrcidIntegration import OperationOrcidIntegration
from code.profdb.Operation3PubInfoscience import OperationPub
from code.profdb.Operation4OpenAlex import OperationOpenAlexID
from code.profdb.Operation5PubOpenAlex import OperationPubOpenAlex
from code.profdb.Operation6Orcid import OperationOrcid
from code.profdb.Operation7Embeddings import OperationEmbedding
from code.profdb.Operation8Representatives import OperationRepresentative
import duckdb
import time
import shutil
import pyarrow as pa
import pyarrow.parquet as pq

class ProfDB:

    def __init__(self, csv_path: str, year_min: int, year_max: int):
        self.db_path = '_temp.duckdb'              # For now, the database is named _temp and will be renamed if the processing is successful
        self.log_path = '_temp.log'
        self.output_path = 'output'
        self.csv_path = csv_path
        self.year_min = year_min
        self.year_max = year_max
        self.logging = self._get_logger()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


    def _get_logger(self):

        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_path)
            ],
            force=True,
        )
        return logging

    def build(self):
        t = time.time()

        OperationProf(self.db_path, self.csv_path).run()
        OperationOrcidIntegration(self.db_path).run()
        OperationPub(self.db_path, self.year_min, self.year_max).run()
        OperationOpenAlexID(self.db_path).run()
        OperationPubOpenAlex(self.db_path, self.year_min, self.year_max).run()
        OperationOrcid(self.db_path).run()
        OperationEmbedding(self.db_path).run()
        OperationRepresentative(self.db_path).run()

        total_time = time.time() - t
        self.logging.info(f"Total processing time: {total_time/60:.2f} minutes")
        self.pack_all(total_time)


    def pack_all(self, tot_time):
        '''
        The Database is renamed with the current date+time+hour, to keep a history.
        The timestamps is saved in a metadata table
        :return:
        '''

        logging.info('Closing and packing the database.')
        logging.info('The logger is being shutdown.')
        logging.shutdown()


        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_table = pd.Series({
            'timestamp': timestamp,
            'total_processing_time_seconds': tot_time
        })

        # put the metadata in the database
        con = duckdb.connect(self.db_path)
        con.register("metadata_df", metadata_table.to_frame().T)
        con.execute("CREATE OR REPLACE TABLE metadata AS SELECT * FROM metadata_df")
        con.close()

        final_folder = f'{self.output_path}/latest'
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(final_folder, exist_ok=True)
        shutil.copy(self.csv_path, f'{final_folder}/input_prof.csv')

        shutil.move(self.db_path, f'{final_folder}/db.duckdb')
        shutil.move(self.log_path, f'{final_folder}/db.log')

        # Copy the latest folder to a timestamped folder
        timestamped_folder = f'{self.output_path}/db_{timestamp}'

        # Make a publication parquet
        con = duckdb.connect(f'{final_folder}/db.duckdb')
        df = con.execute("""
         SELECT sciper,
                title,
                abstract,
                id_pub,
                CASE
                    WHEN id_infoscience IS NOT NULL THEN 'infoscience'
                    ELSE 'openalex'
                    END AS source,
                embedding
         FROM pub
                  INNER JOIN sciper_pub USING (id_pub)
                  INNER JOIN pub_embedding USING (id_pub)
         """).df()
        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        con.close()

        table = pa.Table.from_pandas(df)
        temp_parquet = f'{final_folder}/publications.parquet.tmp'
        final_parquet = f'{final_folder}/publications.parquet'
        pq.write_table(table, temp_parquet)
        shutil.move(temp_parquet, final_parquet)

        # Copy final folder
        shutil.copytree(final_folder, timestamped_folder, dirs_exist_ok=True)

    @abstractmethod
    def _transform(self):
        """Override this in subclasses."""
        pass


