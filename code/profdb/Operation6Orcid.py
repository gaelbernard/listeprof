from code.profdb.Operation0Abstract import OperationAbstract
import requests
import pandas as pd

class OperationOrcid(OperationAbstract):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.orcids = None

    def pre(self):
        try:
            self.orcids = self.con.execute(f"""
with orcid_list as (
    SELECT orcid FROM sciper_openalex WHERE orcid IS NOT NULL
    UNION
    SELECT orcid FROM sciper_orcid_integration
)
select distinct orcid from orcid_list;
""").df().drop_duplicates()['orcid'].tolist()

        except Exception as e:
            self._quit_on_failure(f"Could not retrieve prof scipers from pub and pub_to_prof tables : {e}")

    def trans(self):
        output = []
        links = []  # Combined table for URLs and external identifiers

        for orcid in self.orcids:
            data = self.get_orcid(orcid)

            record_id = {
                'orcid': orcid,
                'claimed': data['history']['claimed'],
                'firstname': data['person']['name'].get('given-names', {}).get('value'),
                'lastname': data['person']['name'].get('family-name', {}).get('value'),
                'other_names': [x.get('content') for x in data['person'].get('other-names', {}).get('other-name', [])],
                'emails': [x.get('email') for x in data['person'].get('emails', {}).get('email', [])],
                'biography': ((data['person'] or {}).get('biography', {}) or {}).get('content'),
                'keyword': ';'.join([x['content'] for x in data['person'].get('keywords', {}).get('keyword', [])]),
            }
            output.append(record_id)

            # Extract researcher URLs
            for url_entry in data['person'].get('researcher-urls', {}).get('researcher-url', []):
                links.append({
                    'orcid': orcid,
                    'type': 'url',
                    'id': url_entry.get('url-name') or url_entry.get('url', {}).get('value'),
                    'url': url_entry.get('url', {}).get('value'),
                })

            # Extract external identifiers
            for ext_id in data['person'].get('external-identifiers', {}).get('external-identifier', []):
                links.append({
                    'orcid': orcid,
                    'type': ext_id.get('external-id-type'),
                    'id': ext_id.get('external-id-value'),
                    'url': ext_id.get('external-id-url', {}).get('value'),
                })

        # Main ORCID table
        orcid_df = pd.DataFrame(output)
        self.con.register("orcid_df", orcid_df)
        self.con.execute("CREATE OR REPLACE TABLE orcid AS SELECT * FROM orcid_df")

        # Links table (URLs + external identifiers)
        links_df = pd.DataFrame(links)
        self.con.register("links_df", links_df)
        self.con.execute("CREATE OR REPLACE TABLE orcid_links AS SELECT * FROM links_df")

    def _orcid_normalize(self, orcid):
        if orcid.startswith('https://orcid.org/'):
            orcid = orcid.split('https://orcid.org/')[-1]
        if orcid.count('-') != 3:
            self._quit_on_failure(f"Invalid ORCID: {orcid}")
        return orcid

    def get_orcid(self, orcid):
        orcid = self._orcid_normalize(orcid)
        url = f"https://pub.orcid.org/v3.0/{orcid}"
        r = requests.get(url, headers={"Accept": "application/json", "User-Agent": "EPFL-ResearchAnalytics (mailto:gael.bernard@epfl.ch)"}, timeout=20)
        r.raise_for_status()
        if r.status_code != 200:
            self._quit_on_failure(f"ORCID API request failed for ORCID: {orcid} with status code {r.status_code}")
        return r.json()


if __name__ == "__main__":
    CSV = "../../pipeline/input/List of professors (Gaël_labList incl. SPC).csv"
    DB = "/Users/gaeberna/EPFL-local/2025-10-08-listProf/output/db_20251203_111132/db.duckdb"
    OperationOrcid(DB).run()