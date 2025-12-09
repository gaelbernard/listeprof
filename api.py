import duckdb
import pandas as pd
from collections import defaultdict
from pathlib import Path
import unicodedata
import re
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from code.profdb.utils import normalize_name
from rapidfuzz import fuzz
from code.profdb.EmbeddingService import EmbeddingService

from dotenv import load_dotenv
from pathlib import Path
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(Path(PROJECT_ROOT) / ".env")
app = FastAPI(title="Prof API")

DB_PATH = "output/latest/db.duckdb"


def get_connection():
    """Get a read-only connection to the database."""
    if not Path(DB_PATH).exists():
        raise FileNotFoundError("Database not found")
    return duckdb.connect(DB_PATH, read_only=True)

@app.get("/profs", summary="List all EPFL professors")
def get_profs():
    """Get list of all professors with their labs, publications, and identifiers extracted from Infoscience, OpenAlex, and Orcid"""
    try:
        con = get_connection()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Database not ready")

    try:
        # Base person info
        df = con.execute("""
                         SELECT sciper, email, lastname, firstname, class_acc
                         FROM prof
                         """).df()

        # Labs information
        labs = con.execute("""
                           SELECT sciper, cf, role, unit_name
                           FROM sciper_lab
                                    INNER JOIN lab USING (cf)
                           """).df()
        labs_grouped = (
            labs.groupby('sciper')
            .apply(lambda x: x[['cf', 'role', 'unit_name']].to_dict('records'), include_groups=False)
            .reset_index(name='labs')
        )
        df = df.merge(labs_grouped, on='sciper', how='left')
        df['n_labs'] = df['labs'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        # Publication counts
        pub_counts = con.execute("""
                                 SELECT p.sciper,
                                        SUM(CASE WHEN pub.id_infoscience IS NOT NULL THEN 1 ELSE 0 END) AS count_infoscience_pubs,
                                        SUM(CASE WHEN pub.openalex_id IS NOT NULL THEN 1 ELSE 0 END)    AS count_openalex_pubs,
                                        COUNT(pub.id_pub)                                               AS count_total_pubs
                                 FROM sciper_pub p
                                          INNER JOIN pub ON p.id_pub = pub.id_pub
                                 GROUP BY p.sciper
                                 """).df()
        df = df.merge(pub_counts, on='sciper', how='left').fillna(0)

        for col in ['count_infoscience_pubs', 'count_openalex_pubs', 'count_total_pubs']:
            df[col] = df[col].astype(int)

        # Identifiers
        identifiers = con.execute("""
                                  WITH all_ids AS (SELECT sciper,
                                                          openalex_id                 AS id,
                                                          openalex_id                 AS url,
                                                          'openalex'                  AS source,
                                                          'doi_infoscience->openalex' AS method
                                                   FROM sciper_openalex
                                                   UNION ALL
                                                   SELECT sciper,
                                                          orcid                              AS id,
                                                          orcid                              AS url,
                                                          'orcid'                            AS source,
                                                          'doi_infoscience->openalex->orcid' AS method
                                                   FROM sciper_openalex
                                                   WHERE orcid IS NOT NULL
                                                   UNION ALL
                                                   SELECT s.sciper,
                                                          o.id,
                                                          o.url,
                                                          o.type                                    AS source,
                                                          'doi_infoscience->openalex->orcid->links' AS method
                                                   FROM orcid_links o
                                                            INNER JOIN sciper_openalex s USING (orcid)
                                                   WHERE orcid IS NOT NULL
                                                   UNION ALL
                                                   SELECT s.sciper,
                                                          o.id,
                                                          o.url,
                                                          o.type                                    AS source,
                                                          'doi_infoscience->openalex->orcid->links' AS method
                                                   FROM orcid_links o
                                                            INNER JOIN sciper_orcid_integration s USING (orcid)
                                                   WHERE orcid IS NOT NULL),
                                       dedup AS (SELECT sciper,
                                                        id,
                                                        url,
                                                        source,
                                                        method,
                                                        ROW_NUMBER() OVER (PARTITION BY sciper, id, url, source ORDER BY method) AS rn
                                                 FROM all_ids)
                                  SELECT sciper, id, url, source, method
                                  FROM dedup
                                  WHERE rn = 1;
                                  """).df().to_dict(orient='records')

        identifiers_by_sciper = defaultdict(list)
        for item in identifiers:
            identifiers_by_sciper[item['sciper']].append({
                'id': item['id'],
                'url': item['url'],
                'source': item['source'],
                'method': item.get('method', '')
            })

        df['identifiers'] = df['sciper'].map(identifiers_by_sciper).apply(lambda x: x if x else [])




        # Alternative names
        query = '''
    SELECT DISTINCT sciper, name FROM (
        SELECT sciper, lastname AS name FROM prof
        UNION
        SELECT sciper, CONCAT(firstname, ' ', lastname) AS name FROM prof
        UNION
        SELECT sciper, CONCAT(lastname, ' ', firstname) AS name FROM prof
        UNION
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

        df = df.merge(names, on='sciper', how='left').rename({'name':'alternative_names'}, axis=1)

        df = df.where(pd.notnull(df), None)


        output = {
            'meta': {
                'count': df.shape[0],
                'last update': con.execute("""SELECT timestamp FROM metadata """).df()['timestamp'].tolist()[0],
            },
            'results': df.to_dict(orient='records')
        }

        return output
    finally:
        con.close()

@app.get("/profs/{sciper}", summary="Retrieve specific EPFL professor")
def get_prof(sciper: int = PathParam(..., examples=[105782], description="SCIPER of the professor (e.g., you can try with 105782)")):
    """Get a single person by sciper."""
    persons = get_profs()
    for p in persons['results']:
        if p['sciper'] == sciper:
            return p
    raise HTTPException(status_code=404, detail="Person not found")

@app.get("/profs/name/{fullname}", summary="Search a professor by their fullname using a fuzzy match. Always returns the top 5 results (sorted by relevance)")
def search_prof(fullname: str = PathParam(..., examples=['Pascal Frossard'], description="name of the professor.")):
    profs = get_profs()
    profs = {x['sciper']: x for x in profs['results']}

    query = normalize_name(fullname)
    scores = []

    for sciper, prof in profs.items():
        best_score = max(
            fuzz.partial_ratio(query, normalize_name(name))
            for name in prof['alternative_names']
        )
        if best_score >= 60:  # threshold out of 100
            scores.append({'sciper': sciper, 'score': best_score})

    if not scores:
        raise HTTPException(status_code=404, detail="Not found")

    scores.sort(key=lambda x: x['score'], reverse=True)

    output = []
    for item in scores[:5]:
        prof = profs[item['sciper']]
        prof['match_score'] = item['score']
        output.append(prof)

    return {'results': output}


@app.get("/profs/cf/{cf}", summary="Get all professors from a lab by its CF identifier")
def search_prof_by_cf(cf: int = PathParam(..., examples=[929], description="CF identifier of the lab.")):
    """Get all professors belonging to a specific lab."""
    profs = get_profs()

    output = []
    for prof in profs['results']:
        for lab in prof.get('labs', []):
            if lab['cf'] == cf:
                output.append(prof)
                break  # avoid duplicates if prof has same cf multiple times

    if not output:
        raise HTTPException(status_code=404, detail=f"No professors found for CF {cf}")

    return {'results': output, 'count': len(output)}

from enum import Enum
import numpy as np

class SearchMethod(str, Enum):
    mean = "mean"
    recency = "recency"
    cluster = "cluster"
    mixed = "mixed"


# Initialize at startup (add this near your other initializations)
embedding_service = EmbeddingService(cache_path="cache_embeddings.lmdb")


def get_representatives():
    """Load all representative embeddings from the database."""
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT sciper, type, weight, embedding
        FROM representative
    """).df()
    con.close()
    return df


@app.get("/search/profs/topic/{topic}", summary="Search professors by research topic using semantic similarity. Returns top 5 results.")
def search_prof_by_topic(
    topic: str = PathParam(..., examples=['machine learning', 'network security', 'quantum computing'], description="Research topic to search"),
    method: SearchMethod = Query(default=SearchMethod.mixed, description="Search method: mean, recency, cluster, or mixed (weighted average of all three)")
):
    """Find professors whose research is most similar to the given topic."""

    profs = get_profs()
    profs_by_sciper = {x['sciper']: x for x in profs['results']}

    # Embed the query
    query_embedding = embedding_service.embed(topic)

    # Load representatives
    representatives = get_representatives()

    # Calculate scores per professor
    scores_by_sciper = {}

    for row in representatives.itertuples():
        sciper = row.sciper
        rep_type = row.type
        rep_embedding = np.array(row.embedding)

        # Euclidean distance (lower is better)
        distance = np.linalg.norm(query_embedding - rep_embedding)

        if sciper not in scores_by_sciper:
            scores_by_sciper[sciper] = {'mean': np.inf, 'recency': np.inf, 'cluster': np.inf}

        if rep_type == 'cluster':
            # Minimum distance across clusters (closest research area)
            scores_by_sciper[sciper]['cluster'] = min(scores_by_sciper[sciper]['cluster'], distance)
        else:
            scores_by_sciper[sciper][rep_type] = distance

    # Compute final score based on method
    final_scores = []
    for sciper, type_scores in scores_by_sciper.items():
        if sciper not in profs_by_sciper:
            continue

        if method == SearchMethod.mixed:
            score = (type_scores['mean'] + type_scores['recency'] + type_scores['cluster']) / 3
        else:
            score = type_scores[method]

        final_scores.append({'sciper': sciper, 'score': score})

    if not final_scores:
        raise HTTPException(status_code=404, detail="No professors found")

    # Sort by score ascending (lower distance = better match)
    final_scores.sort(key=lambda x: x['score'])

    # Build output
    output = []
    for item in final_scores[:5]:
        prof = profs_by_sciper[item['sciper']]
        prof['match_score'] = round(item['score'], 4)
        output.append(prof)

    return {'results': output, 'count': len(output)}

if __name__ == "__main__":
    profs = get_profs()
    profs = {x['sciper']: x for x in profs['results']}

    print (search_prof_by_topic('machine learning', method=SearchMethod.mixed))
    exit()
