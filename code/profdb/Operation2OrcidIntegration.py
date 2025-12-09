import os, logging, duckdb
from code.profdb.Operation0Abstract import OperationAbstract
from code.profdb.utils import *
import pandas as pd
import time
import requests

# -------- concrete implementation --------
class OperationOrcidIntegration(OperationAbstract):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.MAX_RETRIES = 3
        self.WAIT_SECONDS = 60
        self.orcid = None
        self.scipers = None

    def pre(self):
        try:
            self.orcid = self.SciperToOrcidIntegration()
        except Exception as e:
            self._quit_on_failure(f"Could not connect to Orcid Integration : {e}")

        # Convert to DataFrame
        try:
            self.orcid = pd.DataFrame([{'sciper': int(k), 'orcid': v['orcid']} for k, v in self.orcid.items() if k.isdigit()])
        except Exception as e:
            self._quit_on_failure(f"Could not process ORCID data : {e}")
        try:
            self.scipers = set(self.con.execute("SELECT distinct sciper FROM prof").df()["sciper"].tolist())
        except Exception as e:
            self._quit_on_failure(f"Could not retrieve prof scipers from prof table : {e}")

    def SciperToOrcidIntegration(self):
        # --- Retry loop ---
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                url = "https://orcid-integration.epfl.ch/export/"
                response = requests.get(url)
                response.raise_for_status()  # Raise an error for bad responses
                return response.json()

            except Exception as e:
                time.sleep(self.WAIT_SECONDS)
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to connect to ORCID integration after {self.MAX_RETRIES} attempts: {e}") from e


    def trans(self):

        # Keep only orcid of profs
        self.orcid = self.orcid[self.orcid['sciper'].isin(self.scipers)]

        # Create table
        self.con.register("orcid_df", self.orcid)
        self.con.execute("CREATE OR REPLACE TABLE sciper_orcid_integration AS SELECT * FROM orcid_df")

        # Add index
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_sciper_orcid_sciper ON sciper_orcid_integration (sciper)")
        logging.info(f"Successfully retrieved {self.orcid.shape[0]} ORCID for profs")

# -------- run --------
if __name__ == "__main__":
    DB = "prof_new2.duckdb"
    OperationOrcidIntegration(DB).run()