import os, logging, duckdb
from code.profdb.Operation0Abstract import OperationAbstract
from code.profdb.utils import *
import pandas as pd
from dspace_rest_client.client import DSpaceClient

# -------- concrete implementation --------
class OperationPub(OperationAbstract):
    def __init__(self, db_path: str, year_min: int, year_max: int):
        super().__init__(db_path)
        self.query = f'{year_min}-01-01 TO {year_max}-12-31'
        if self.sample:
            self.query = f'2025-01-01 TO 2025-04-01'
        self.infoscience = None
        self.sciper_profs = None

    def pre(self):
        try:
            self.infoscience = self.connect_to_infoscience()
        except Exception as e:
            self._quit_on_failure(f"Could not connect to Infoscience DSpace API : {e}")
        try:
            self.sciper_profs = set(self.con.execute("SELECT distinct sciper FROM prof").df()["sciper"].tolist())
        except Exception as e:
            self._quit_on_failure(f"Could not retrieve prof scipers from prof table : {e}")

    def trans(self):
        pub = []
        sciper_pub = []
        dsos = self.infoscience.search_objects(query=f'dc.date.issued:[{self.query}] AND dspace.entity.type:publication', configuration='researchoutputs')
        id_pub = 0
        for item in dsos:
            r = item.as_dict().get("metadata", {})
            authors_sciper = set([int(sciper["value"]) for sciper in r.get("cris.virtual.sciperId") or [] if sciper.get("value") and sciper["value"].isdigit()])
            profs_sciper = authors_sciper.intersection(self.sciper_profs)
            if not profs_sciper:
               continue
            id_pub += 1

            pub.append({
                "id_pub": id_pub,
                "id_infoscience": (r.get("dc.identifier.uri") or [{}])[0].get("value"),
                "doi": (r.get("dc.identifier.doi") or [{}])[0].get("value"),
                "title": (r.get("dc.title") or [{}])[0].get("value"),
                "abstract": (r.get("dc.description.abstract") or [{}])[0].get("value"),
                #"type": (r.get("dspace.entity.type") or [{}])[0].get("value"),
                "date_issued": (r.get("dc.date.issued") or [{}])[0].get("value"),
            })
            for sciper in profs_sciper:
                if sciper in profs_sciper:
                    sciper_pub.append({
                        "id_pub": id_pub,
                        "sciper": sciper,
                    })

        if len(pub) == 0:
            self._quit_on_failure(f"No publications found in Infoscience")
        if len(sciper_pub) == 0:
            self._quit_on_failure(f"No prof publications found in Infoscience")

        pub_df = pd.DataFrame(pub)
        sciper_pub_df = pd.DataFrame(sciper_pub)

        pub_df['doi'] = pub_df['doi'].apply(normalize_doi)
        pub_df['year_issued'] = self.normalize_publication_date(pub_df['date_issued'])
        pub_df = pub_df.drop(columns=['date_issued'])

        self.con.register("sciper_pub_df", sciper_pub_df)
        self.con.execute("CREATE OR REPLACE TABLE sciper_pub AS SELECT * FROM sciper_pub_df")
        logging.info(f"Successfully built sciper_pub table with {len(sciper_pub_df)} prof-publication associations")

        self.con.register("pub_df", pub_df)
        self.con.execute("CREATE OR REPLACE TABLE pub AS SELECT * FROM pub_df")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_pub_id_pub ON pub (id_pub)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_sciper_pub_sciper ON sciper_pub (sciper)")
        logging.info(f"Successfully built pub table with {len(pub_df)} publications")


    def normalize_publication_date(self, series: pd.Series) -> pd.Series:
        """
        Best-effort date parser:
          1) Parse with pandas (ISO etc.)
          2) Retry with dayfirst=True (handles 01.05.2025, 01-05-2025)
          3) Fallback to just the 4-digit year (sets to Jan 1)
        Returns a pandas Series of dtype datetime64[ns] with NaT where parsing failed.
        """
        s = series.copy()

        # Pass 1: normal parse (handles 'YYYY-MM-DD', 'YYYY-MM', 'YYYY')
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")

        # Pass 2: retry remaining with dayfirst (handles '01.05.2025', '01-05-2025')
        mask = parsed.isna() & s.notna()
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(s[mask], errors="coerce", format="mixed", dayfirst=True)

        # Pass 3: fallback to year-only if we can extract a 4-digit year
        mask = parsed.isna() & s.notna()
        if mask.any():
            years = s[mask].astype(str).str.extract(r"\b((?:19|20)\d{2})\b")[0]
            year_idx = years.dropna().index
            if len(year_idx) > 0:
                parsed.loc[year_idx] = pd.to_datetime(years.loc[year_idx], format="%Y", errors="coerce")

        return parsed.dt.year.astype('Int64')


    def connect_to_infoscience(self):
        d = DSpaceClient()
        d.authenticate()
        return d


# -------- run --------
if __name__ == "__main__":
    DB = "../../temp.duckdb"
    year_min = 2025
    year_max = 2025
    OperationPub(DB, year_min, year_max).run()