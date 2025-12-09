import duckdb
import pandas as pd
import unicodedata
import re

import unicodedata

def normalize_name(s: str) -> str:
    if not isinstance(s, str) or s.strip() == "":
        return ""

    # 1. Normalize accents (NFD splits letters + diacritics)
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')

    # 2. Replace unwanted punctuation with a space
    s = re.sub(r"[-,.;:'’_]", " ", s)

    # 3. Lowercase
    s = s.lower()

    # 4. Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)

    # 5. Trim
    return s.strip()

duck_db_path = "../output/db_20251203_194614/db.duckdb"
con = duckdb.connect(duck_db_path, read_only=True)


query = '''
SELECT DISTINCT sciper, name FROM (
    SELECT sciper, CONCAT(lastname, ' ', firstname) AS name
    FROM orcid
    INNER JOIN sciper_orcid_integration USING (orcid)
    UNION
    SELECT sciper, CONCAT(firstname, ' ', lastname) AS name
    FROM orcid
    INNER JOIN sciper_orcid_integration USING (orcid)
    UNION
    SELECT sciper, CONCAT(firstname, ' ', lastname) AS name
    FROM orcid
    INNER JOIN sciper_openalex USING (orcid)
    UNION
    SELECT sciper, CONCAT(lastname, ' ', firstname) AS name
    FROM orcid
    INNER JOIN sciper_openalex USING (orcid)
    UNION
    SELECT sciper, display_name AS name
    FROM sciper_openalex
    UNION
    SELECT sciper, n AS name
    FROM (
        SELECT sciper, UNNEST(other_names) AS n
        FROM orcid
        INNER JOIN sciper_openalex USING (orcid)
        WHERE other_names IS NOT NULL
    )
    UNION
    SELECT sciper, n AS name
    FROM (
        SELECT sciper, UNNEST(other_names) AS n
        FROM orcid
        INNER JOIN sciper_orcid_integration USING (orcid)
        WHERE other_names IS NOT NULL
    )

    -- 6. UNNEST display_name_alternatives (array)
    UNION

    SELECT sciper, n AS name
    FROM (
        SELECT sciper, UNNEST(display_name_alternatives) AS n
        FROM sciper_openalex
        WHERE display_name_alternatives IS NOT NULL
    )
);
'''
data = con.execute(query).df()
data['name'] = data['name'].str.lower()
data['name'] = data['name'].apply(normalize_name)
data = data.drop_duplicates()
names = data.groupby('sciper')['name'].agg(list)
print (names)
exit()
for _,row in df.iterrows():
    print (sciper, sciper_name)
    exit()
df = con.execute(query).df()
print (df)
exit()
#
#
#

print (df.shape)