import os, logging, duckdb
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from code.profdb.Operation0Abstract import OperationAbstract
import pandas as pd


# -------- concrete implementation --------
class OperationProf(OperationAbstract):
    def __init__(self, db_path: str, csv_path: str):
        super().__init__(db_path)
        self.csv_path = csv_path
        self.master_list = None

    def pre(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Missing file: {self.csv_path}")

        try:
            self.master_list = pd.read_csv(self.csv_path)
        except Exception as e:
            self._quit_on_failure(f"Could not read CSV file {self.csv_path}: {e}")

        if self.sample:
            self.master_list = self.master_list.head(20)

    def trans(self):
        if not os.path.exists(self.csv_path):
            self._quit_on_failure(f"File {self.csv_path} does not exist.")

        master_list = self.master_list
        columns_needed = [
            "N° sciper",
            "email interne unique (1par ligne)",
            "Nom Sciper",
            "Prénom Sciper",
            "date création",
            "classe acc.",
            "CF unité",
            "Nom unité",
            "fonction acc.",
        ]
        missing = set(columns_needed) - set(master_list.columns)
        if missing:
            self._quit_on_failure(f"File {self.csv_path} does not have the required columns: {missing}")

        logging.info(f"Validated master list: {len(master_list)} rows")

        # Build prof table
        prof = (
            master_list.rename(columns={
                "N° sciper": "sciper",
                "email interne unique (1par ligne)": "email",
                "Nom Sciper": "lastname",
                "Prénom Sciper": "firstname",
                "date création": "creation_date",
                "classe acc.": "class_acc"
            })[["sciper", "email", "lastname", "firstname", "creation_date", "class_acc"]]
            .drop_duplicates()
            .sort_values("firstname")
        )
        prof["creation_date"] = pd.to_datetime(prof["creation_date"], format="%Y/%m/%d %H:%M:%S")

        # Build lab table
        lab = (
            master_list.rename(columns={
                "CF unité": "cf",
                "Nom unité": "unit_name"
            })[["cf", "unit_name"]]
            .drop_duplicates()
        )

        # Build sciper_lab table
        sciper_lab = (
            master_list.rename(columns={
                "N° sciper": "sciper",
                "CF unité": "cf",
                "fonction acc.": "role"
            })[["sciper", "cf", "role"]]
            .dropna()
            .drop_duplicates()
        )

        # Insert into DuckDB
        self.con.register("prof_df", prof)
        self.con.execute("CREATE OR REPLACE TABLE prof AS SELECT * FROM prof_df")
        logging.info(f"Successfully built prof table: {len(prof)} rows")

        self.con.register("lab_df", lab)
        self.con.execute("CREATE OR REPLACE TABLE lab AS SELECT * FROM lab_df")
        logging.info(f"Successfully built lab table: {len(lab)} rows")

        self.con.register("sciper_lab_df", sciper_lab)
        self.con.execute("CREATE OR REPLACE TABLE sciper_lab AS SELECT * FROM sciper_lab_df")
        logging.info(f"Successfully built sciper_lab table: {len(sciper_lab)} rows")

        # Create indexes
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_prof_sciper ON prof (sciper)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_lab_cf ON lab (cf)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_sciper_lab_sciper ON sciper_lab (sciper)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_sciper_lab_cf ON sciper_lab (cf)")


# -------- run --------
if __name__ == "__main__":
    CSV = "../../pipeline/input/List of professors (Gaël_labList incl. SPC).csv"
    DB = "../../temp.duckdb"
    OperationProf(DB, CSV).run()