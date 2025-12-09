import os, logging, duckdb
from code.profdb.Operation0Abstract import OperationAbstract
import time
import re
import unicodedata
import requests
import pandas as pd
from more_itertools import chunked
from rapidfuzz.distance import Levenshtein
from dotenv import load_dotenv
load_dotenv()

MAX_BATCH_SIZE = 40  # keep URL under control
MAX_RETRIES = 3  # stop after 3 tries, as requested
RETRY_DELAY = 60
MAILTO = 'gael.bernard@epfl.ch'
session = requests.Session()
session.headers.update({
    "User-Agent": f"EPFL-ResearchAnalytics (mailto:{MAILTO})",
    "Accept": "application/json",
})
timeout = 30

api_key_openalex = os.getenv('API_KEY_OPENALEX')

# -------- concrete implementation --------
class OperationOpenAlexID(OperationAbstract):
    def __init__(self, db_path: str, max_doi: int = 50):
        super().__init__(db_path)
        self.max_doi = max_doi
        self.sciper_dois_name = None

    def pre(self):

        # retrieve list of prof scipers from prof table
        try:
            df = self.con.execute(f"SELECT sciper, doi, firstname, lastname FROM sciper_pub JOIN pub USING (id_pub) JOIN prof USING (sciper) WHERE doi IS NOT NULL ORDER BY year_issued DESC").df().drop_duplicates()
            df['name'] = df['firstname'].fillna('') + ' ' + df['lastname'].fillna('')
            df = df.groupby('sciper').agg(
                total_dois=('doi', 'count'),
                name=('name', 'first'),
                dois=('doi', list),
            ).reset_index()
            # limit to max_doi
            df['dois'] = [x[:self.max_doi] for x in df['dois'].tolist()]
            self.sciper_dois_name = df

        except Exception as e:
            self._quit_on_failure(f"Could not retrieve prof scipers from pub and pub_to_prof tables : {e}")

        if self.sample:
            self.sciper_dois_name = self.sciper_dois_name.head(5)

    def trans(self):

        # Step 1: infer openalex ids from dois
        self.sciper_dois_name['name_clean'] = self.sciper_dois_name['name'].apply(self.standardize_name)
        prof_openalex_ids = []
        for _, row in self.sciper_dois_name.iterrows():
            sciper = row['sciper']
            openalexIDs = self.dois_to_openalex_ids(row['name'], row['dois'])
            for oaid in openalexIDs:
                prof_openalex_ids.append({
                    'sciper': sciper,
                    'openalex_id': oaid
                })
        prof_openalex_ids = pd.DataFrame(prof_openalex_ids)
        # Apply manual correction
        corrections = pd.DataFrame([
            {'sciper':317488, 'openalex_id':'https://openalex.org/A5025196309'}, #Mackenzie Mathis
            {'sciper':318514, 'openalex_id':'https://openalex.org/A5013861606'}, #Alexander Mathis
        ])
        # Remove from df mistakes (entries for scipers that have manual corrections)
        prof_openalex_ids = prof_openalex_ids[~prof_openalex_ids['sciper'].isin(corrections['sciper'])]
        prof_openalex_ids = pd.concat([prof_openalex_ids, corrections], ignore_index=True)

        logging.info(f"Successfully retrieved {prof_openalex_ids.shape[0]} OpenAlex IDs for profs")

        details = pd.DataFrame(self.get_openalex_details(prof_openalex_ids['openalex_id'].tolist()))

        # Adding the sciper info
        details = prof_openalex_ids.merge(details, left_on='openalex_id', right_on='id', how='inner')
        details = details.drop(columns=['id'])

        logging.info(f"Successfully retrieved {details.shape[0]} OpenAlex authors information")

        self.con.register("details", details)
        self.con.execute("CREATE OR REPLACE TABLE sciper_openalex AS SELECT * FROM details")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_prof_openalex_sciper ON sciper_openalex (sciper)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_prof_openalex_openalex_id ON sciper_openalex (openalex_id)")

    def standardize_name(self, name):
        if not name or not isinstance(name, str):
            return None
        s = name.strip()
        if "," in s:
            last, first = [p.strip() for p in s.split(",", 1)]
            s = f"{first} {last}"
        nfkd = unicodedata.normalize("NFKD", s)
        clean = nfkd.encode("ASCII", "ignore").decode()
        parts = re.sub(r"[^\w\s-]", "", clean).split()
        if len(parts) < 2:
            return None
        first, last = parts[0], parts[-1]
        initials = "".join(p[0] for p in first.split("-") if p) or first[0]
        return f"{initials.lower()} {last.lower()}"

    def dois_to_openalex_ids(self, display_name_source, dois, max_edit_distance=1):
        """
        For each person: from a set of DOIs -> get OpenAlex works -> collect candidate OpenAlex author IDs,
        filter by name similarity against the source display name, and return the likely IDs.
        """

        # Normalize DOIs and drop obvious junk
        clean_set = {d for d in (dois or [])}
        clean_set.discard("")  # remove blanks

        authors = []
        publication_found_count = 0

        for doi_chunk in chunked(sorted(clean_set), MAX_BATCH_SIZE):
            # Build request with params so requests handles encoding
            params = {
                "filter": f"doi:{'|'.join(doi_chunk)}",
                "select": "authorships",
                "per_page": MAX_BATCH_SIZE,                # OK for works endpoint
                "mailto": MAILTO,
                "api_key": api_key_openalex
            }
            url = "https://api.openalex.org/works"

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = session.get(url, params=params, timeout=timeout)
                    if resp.status_code in (403, 429):
                        time.sleep(RETRY_DELAY)
                        continue
                    resp.raise_for_status()
                    success = True
                    break
                except requests.RequestException as e:
                    if attempt == MAX_RETRIES:
                        raise RuntimeError(f"Failed to connect to OpenAlex after {MAX_RETRIES} attempts: {e}") from e
                    else:
                        time.sleep(RETRY_DELAY)

            if not success:
                # skip this batch; continue to next
                continue

            data = resp.json()
            for pub in data.get("results", []) or []:
                publication_found_count += 1
                for a in pub.get("authorships", []) or []:
                    author_obj = a.get("author") or {}
                    if author_obj.get("id"):    # only keep authors with an OpenAlex ID
                        authors.append(author_obj)

        if not authors:
            return []

        # Aggregate by author ID
        df = (
            pd.DataFrame(authors)
              .groupby("id")
              .agg(display_name=("display_name", "first"), count=("id", "size"))
              .sort_values("count", ascending=False)
              .head(50)  # keep a reasonable candidate set
              .reset_index()
        )

        # Name similarity filter
        df["display_name_std"] = df["display_name"].apply(self.standardize_name)
        name_std = self.standardize_name(display_name_source)
        if name_std is None:
            # If we can't standardize the source name, just return the top few author IDs
            return df["id"].tolist()[:5]

        df = df.dropna(subset=["display_name_std"])
        if df.empty:
            return []

        df["edit_distance"] = df["display_name_std"].apply(lambda s: Levenshtein.distance(s, name_std))
        df = df.sort_values(["edit_distance", "count"], ascending=[True, False])

        # Keep only very close matches by default
        close = df[df["edit_distance"] <= max_edit_distance]
        if not close.empty:
            return close["id"].tolist()

        # If no tight match, fall back to the top 3 candidates
        return []

    def get_openalex_details(self, list_openAlex_ids, MAX_BATCH_SIZE=50):

        output = []
        for batch_openAlex_ids in chunked(list_openAlex_ids, MAX_BATCH_SIZE):
            if api_key_openalex:
                url = f"https://api.openalex.org/authors?filter=id:{'|'.join(batch_openAlex_ids)}&api_key={api_key_openalex}"
            else:
                url = f"https://api.openalex.org/authors?filter=id:{'|'.join(batch_openAlex_ids)}"
            i = 0
            while True:
                try:
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an error for bad responses
                    break  # Exit the loop if the request was successful
                except requests.RequestException as e:
                    i += 1
                    if i > MAX_RETRIES:
                        raise RuntimeError(f"Failed to connect to OpenAlex after {MAX_RETRIES} attempts: {e}") from e
                    time.sleep(RETRY_DELAY)
            results = response.json().get('results') or []
            for author in results:
                output.append(author)

        return output


# -------- run --------
if __name__ == "__main__":
    DB = "../../temp.duckdb"
    #2024-12-01 TO 2024-12-03
    OperationOpenAlexID(DB).run()