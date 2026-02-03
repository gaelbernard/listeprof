import os, logging, duckdb
from datetime import datetime
from code.profdb.Operation0Abstract import OperationAbstract
from more_itertools import chunked
from rapidfuzz.distance import Levenshtein
import requests
from code.profdb.utils import *
import unicodedata
from dotenv import load_dotenv
load_dotenv()

api_key_openalex = os.getenv('API_KEY_OPENALEX')


# -------- concrete implementation --------
class OperationPubOpenAlex(OperationAbstract):
    def __init__(self, db_path: str, year_min, year_max):
        super().__init__(db_path)
        self.year_min = year_min
        self.year_max = year_max
        self.sciper_openalex_id = None

    def pre(self):
        self.sciper_openalex_id = self.con.execute(
            "SELECT sciper, openalex_id FROM sciper_openalex"
        ).df().drop_duplicates()

        # Add a column openalex_id to table pub
        try:
            self.con.execute("ALTER TABLE pub ADD COLUMN IF NOT EXISTS openalex_id STRING")
        except Exception as e:
            self._quit_on_failure(f"Could not add openalex_id column to pub table : {e}")

    def _retrieve_openalex_pub(self, author_id: str) -> list:
        """Retrieve all publications for an OpenAlex author ID."""
        cursor = '*'
        results = []

        while cursor:
            url = (
                f'https://api.openalex.org/works?'
                f'filter=author.id:{author_id},'
                f'publication_year:{self.year_min}-{self.year_max},'
                f'type:article|book|book-chapter'
                f'&include_xpac=true&cursor={cursor}&per-page=50'
            )
            if api_key_openalex:
                url += f'&api_key={api_key_openalex}'

            response = requests.get(url)
            if response.status_code != 200:
                logging.warning(f"OpenAlex API error {response.status_code} for author {author_id}")
                break

            data = response.json()
            results.extend(data.get('results') or [])
            cursor = data.get('meta', {}).get('next_cursor')

        return results

    def basic_data_cleanup(self, text: str) -> str:
        """Remove accented characters, multiple spaces, leading/trailing spaces."""
        if not text:
            return ''
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _retrieve_infoscience_pub(self, sciper: int):
        """Retrieve existing publications for a professor from the database."""
        query = '''
            SELECT pub.* FROM pub 
            INNER JOIN sciper_pub USING (id_pub)
            WHERE sciper = ?
        '''
        return self.con.execute(query, [sciper]).df()

    def reverse_abstract_index(self, abstract_inverted_index: dict) -> str:
        """Convert OpenAlex inverted abstract index back to plain text."""
        if not abstract_inverted_index:
            return ''
        words = []
        for word, positions in abstract_inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        words.sort()
        return ' '.join(word for _, word in words)

    def _find_matching_pub(self, existing_pubs, doi: str, title_clean: str):
        """Find a matching publication by DOI or title similarity."""
        if existing_pubs.empty:
            return None

        # Check DOI match first
        if doi:
            doi_match = existing_pubs[existing_pubs['doi'] == doi]
            if not doi_match.empty:
                return doi_match.iloc[0]

        # Check title similarity
        for _, row in existing_pubs.iterrows():
            if Levenshtein.distance(row['title'], title_clean) < 3:
                return row

        return None

    def trans(self):
        max_id_result = self.con.execute("SELECT MAX(id_pub) as max_id FROM pub").df().iloc[0]['max_id']
        next_id = int(max_id_result) + 1 if max_id_result is not None else 1

        for _, row in self.sciper_openalex_id.iterrows():
            # Convert numpy types to native Python types
            sciper = int(row['sciper'])
            openalex_author_id = str(row['openalex_id'])

            # Get existing publications for this professor
            existing_pubs = self._retrieve_infoscience_pub(sciper)
            existing_pubs['title'] = existing_pubs['title'].apply(self.basic_data_cleanup)

            new_records = []

            for pub in self._retrieve_openalex_pub(openalex_author_id):
                openalex_work_id = pub.get('id')
                doi = normalize_doi(pub.get('doi'))
                title = pub.get('title') or ''
                title_clean = self.basic_data_cleanup(title)
                abstract = self.reverse_abstract_index(pub.get('abstract_inverted_index'))
                year_issued = pub.get('publication_year')

                # Check if publication already exists
                match = self._find_matching_pub(existing_pubs, doi, title_clean)

                if match is not None:
                    existing_id = int(match['id_pub'])

                    # Update pub table with openalex_id
                    self.con.execute(
                        'UPDATE pub SET openalex_id = ? WHERE id_pub = ?',
                        [openalex_work_id, existing_id]
                    )

                    # If the abstract is missing in infoscience, update it
                    if not match['abstract'] and abstract:
                        self.con.execute(
                            'UPDATE pub SET abstract = ? WHERE id_pub = ?',
                            [abstract, existing_id]
                        )
                else:
                    # New publication - add to batch
                    new_records.append({
                        'id_pub': next_id,
                        'doi': doi,
                        'id_infoscience': None,
                        'title': title_clean,
                        'abstract': abstract,
                        'year_issued': int(year_issued) if year_issued else None,
                        'openalex_id': openalex_work_id,
                    })
                    next_id += 1

            # Bulk insert new records
            if new_records:
                df_new_records = pd.DataFrame(new_records)

                # Ensure correct types for DuckDB
                df_new_records['id_pub'] = df_new_records['id_pub'].astype(int)
                df_new_records['year_issued'] = df_new_records['year_issued'].astype('Int64')  # Nullable int

                self.con.register("new_pub_df", df_new_records)
                self.con.execute('''
                    INSERT INTO pub (id_pub, doi, id_infoscience, title, abstract, year_issued, openalex_id)
                    SELECT id_pub, doi, id_infoscience, title, abstract, year_issued, openalex_id FROM new_pub_df
                ''')
                self.con.unregister("new_pub_df")

                # Link new publications to the professor (bulk)
                df_sciper_pub = pd.DataFrame({
                    'sciper': [sciper] * len(new_records),
                    'id_pub': [r['id_pub'] for r in new_records]
                })
                df_sciper_pub['sciper'] = df_sciper_pub['sciper'].astype(int)
                df_sciper_pub['id_pub'] = df_sciper_pub['id_pub'].astype(int)

                self.con.register("new_sciper_pub_df", df_sciper_pub)
                self.con.execute('''
                    INSERT INTO sciper_pub (sciper, id_pub)
                    SELECT sciper, id_pub FROM new_sciper_pub_df
                ''')
                self.con.unregister("new_sciper_pub_df")


# -------- run --------
if __name__ == "__main__":
    DB = "../../temp.duckdb"
    year_min = 2022
    year_max = 2025
    OperationPubOpenAlex(DB, year_min, year_max).run()