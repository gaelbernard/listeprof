import duckdb
import pandas as pd
import unicodedata
import re

def simplify_name(name):
    """
    Simplify an author name to a token of the form "first-initial last-name".
    :param name: name to simplify. It can be in the form "Last, First" or "First Last" or "First-Last".
    :return: simplified token or None if the name is invalid (e.g., too short, no last name).
    """
    # Simplify an author name to "first-initial last-name"
    if not name or not isinstance(name, str):
        return None
    s = name.strip()
    # Convert "Last, First" to "First Last"
    if ',' in s:
        parts = s.split(',', 1)
        if len(parts) == 2:
            last, first = parts[0].strip(), parts[1].strip()
            s = f"{first} {last}"
    # Remove accents and non-ASCII characters
    nfkd = unicodedata.normalize("NFKD", s)
    clean = nfkd.encode("ASCII", "ignore").decode()
    # Keep only letters and spaces
    parts = re.sub(r"[^\w\s]", "", clean).split()
    if len(parts) < 2:
        # If we cannot get at least first and last name parts, consider it invalid for tokenization
        return None
    first, last = parts[0], parts[-1]
    # Use first part's first initial (or first letter of each hyphenated part) and last name
    initials = ''.join(p[0] for p in first.split('-') if p) or first[0]
    token = f"{initials.lower()} {last.lower()}"
    # Store in cache
    return token

duck_db_path = "../output/db_20251203_194614/db.duckdb"
con = duckdb.connect(duck_db_path, read_only=True)

checks = []
checks.append(
    con.execute("""
        SELECT sciper, 
               'epfl' as source1_type, CONCAT(IFNULL(firstname,''), ' ', IFNULL(lastname,'')) as source1_name, sciper as source1_id, 
               'openalex' as source2_type, display_name as source2_name, openalex_id as source2_id, 
        FROM prof
        INNER JOIN sciper_openalex USING (sciper)
    """).df()
)
checks.append(
    con.execute("""
        SELECT sciper, 
               'epfl' as source1_type, CONCAT(IFNULL(prof.firstname,''), ' ', IFNULL(prof.lastname,'')) as source1_name, sciper as source1_id, 
               'orcid integration' as source2_type, CONCAT(IFNULL(orcid.firstname,''), ' ', IFNULL(orcid.lastname,'')) as source2_name, orcid as source2_id, 
        FROM prof
        INNER JOIN sciper_orcid_integration USING (sciper)
        INNER JOIN orcid USING (orcid)
    """).df()
)
checks.append(
    con.execute("""
        SELECT sciper, 
               'openalex' as source1_type, sciper_openalex.display_name as source1_name, openalex_id as source1_id, 
               'orcid' as source2_type, CONCAT(IFNULL(orcid.firstname,''), ' ', IFNULL(orcid.lastname,'')) as source2_name, orcid as source2_id, 
        FROM sciper_openalex
        INNER JOIN orcid USING (orcid)
    """).df()
)
checks = pd.concat(checks).reset_index(drop=True)
checks['source1_name_norm'] = checks['source1_name'].apply(simplify_name)
checks['source2_name_norm'] = checks['source2_name'].apply(simplify_name)
checks = checks[checks['source1_name_norm']!=checks['source2_name_norm']]
print (checks.to_string())
exit()
