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
        # todo: use the openalex key

    def pre(self):

        self.sciper_openalex_id = self.con.execute(f"SELECT sciper, openalex_id FROM sciper_openalex").df().drop_duplicates()

        # Add a column openalex_id to table pub
        try:
            self.con.execute("ALTER TABLE pub ADD COLUMN IF NOT EXISTS openalex_id STRING")
        except Exception as e:
            self._quit_on_failure(f"Could not add openalex_id column to pub table : {e}")

    def _retrieve_openalex_pub(self, id):
        cursor = '*'
        results = []
        while cursor:

            url = f'https://api.openalex.org/works?filter=author.id:{id},publication_year:{self.year_min}-{self.year_max},type:article|book|book-chapter&include_xpac=true&cursor={cursor}&per-page=50'
            if api_key_openalex:
                url += f'&api_key={api_key_openalex}'

            data = requests.get(url).json()
            results.extend(data.get('results') or [])
            try:
                cursor = data['meta']['next_cursor']
            except:
                cursor = None
        return results

    def basic_data_cleanup(self, text):
        # remove accented characters, multiple spaces, leading/trailing spaces
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _retrieve_infoscience_pub(self, sciper):
        query = f'''
        select pub.* from pub 
        INNER JOIN sciper_pub USING (id_pub)
        WHERE sciper = {sciper}
        '''
        # query = f'''
        #         select type, count(*) as nb from pub GROUP BY type
        #         '''
        return self.con.execute(query).df()

    def reverse_abstract_index(self, abstract_inverted_index):
        if not abstract_inverted_index:
            return ''
        # reverse the inverted index
        index_items = abstract_inverted_index.items()
        words = []
        for word, positions in index_items:
            for pos in positions:
                words.append((pos, word))
        words.sort()
        abstract = ' '.join([word for pos, word in words])
        return abstract

    def trans(self):

        max_id = self.con.execute("SELECT MAX(id_pub) as max_id FROM pub").df().iloc[0]['max_id']
        next_id = max_id + 1
        for _, (sciper, id) in self.sciper_openalex_id.iterrows():
            existing_pubs = self._retrieve_infoscience_pub(sciper)
            existing_pubs['title'] = existing_pubs['title'].apply(self.basic_data_cleanup)
            new_records = []
            for pub in self._retrieve_openalex_pub(id):
                id = next_id
                doi = normalize_doi(pub.get('doi'))
                title = pub.get('title') or ''
                title_clean = self.basic_data_cleanup(title)

                abstract = self.reverse_abstract_index(pub.get('abstract_inverted_index'))
                year_issued = pub.get('publication_year')

                # check if publication already exists in infoscience
                match = existing_pubs[
                    (existing_pubs['doi'] == doi) |
                    (Levenshtein.distance(existing_pubs['title'], title_clean) < 3)
                ]
                if not match.empty:
                    first_match = match.iloc[0]
                    existing_id = first_match['id_pub']
                    # Update pub table with openalex_id
                    self.con.execute(f'''
                        UPDATE pub SET openalex_id = '{pub.get('id')}' WHERE id_pub = {existing_id}
                    ''')

                    # if the abstract is missing in infoscience, update it
                    if not first_match['abstract'] and abstract:
                        self.con.execute(f'''
                            UPDATE pub SET abstract = ? WHERE id_pub = {existing_id}
                        ''', (abstract,))
                else:
                    # insert new publication
                    new_records.append({
                        'id_pub': id,
                        'doi': doi,
                        'id_infoscience': None,
                        'title': title_clean,
                        'abstract': abstract,
                        'year_issued': year_issued,
                        'openalex_id': pub.get('id'),
                    })
                    next_id += 1

                # finally, insert new records into pub table
            if new_records:
                df_new_records = pd.DataFrame(new_records)
                self.con.register("new_pub_df", df_new_records)
                self.con.execute('''
                    INSERT INTO pub (id_pub, doi, id_infoscience, title, abstract, year_issued, openalex_id)
                    SELECT id_pub, doi, id_infoscience, title, abstract, year_issued, openalex_id FROM new_pub_df
                ''')

                # Link new publications to the professor (bulk)
                df_sciper_pub = pd.DataFrame({
                    'sciper': [sciper] * len(new_records),
                    'id_pub': [r['id_pub'] for r in new_records]
                })
                self.con.register("new_sciper_pub_df", df_sciper_pub)
                self.con.execute('''
                    INSERT INTO sciper_pub (sciper, id_pub)
                    SELECT sciper, id_pub FROM new_sciper_pub_df
                ''')



# -------- run --------
if __name__ == "__main__":
    DB = "../../temp.duckdb"
    year_min = 2022
    year_max = 2025
    OperationPubOpenAlex(DB, year_min, year_max).run()