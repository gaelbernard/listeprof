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
from enum import Enum
import numpy as np

from dotenv import load_dotenv
from pathlib import Path
from fastapi.responses import RedirectResponse
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(Path(PROJECT_ROOT) / ".env")

app = FastAPI(
    title="Prof API",
    description="API to search and retrieve EPFL professors with their labs, publications, and identifiers from Infoscience, OpenAlex, and ORCID.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to documentation."""
    return RedirectResponse(url="/docs")


DB_PATH = "output/latest/db.duckdb"

# Initialize embedding service at startup
embedding_service = EmbeddingService(cache_path="cache_embeddings.lmdb")


class SearchMethod(str, Enum):
    mean = "mean"
    recency = "recency"
    cluster = "cluster"
    mixed = "mixed"


def get_connection():
    """Get a read-only connection to the database."""
    if not Path(DB_PATH).exists():
        raise FileNotFoundError("Database not found")
    return duckdb.connect(DB_PATH, read_only=True)


def build_response(profs: list[dict], last_update: str) -> dict:
    """
    Build a standardized API response with enriched meta information.

    Args:
        profs: List of professor dictionaries
        last_update: Timestamp of last database update

    Returns:
        Standardized response dict with meta and results
    """
    # Calculate aggregate statistics
    total_infoscience = sum(p.get('count_infoscience_pubs', 0) or 0 for p in profs)
    total_openalex = sum(p.get('count_openalex_pubs', 0) or 0 for p in profs)
    total_pubs = sum(p.get('count_total_pubs', 0) or 0 for p in profs)

    # Count unique identifiers across all profs
    unique_identifiers = set()
    for p in profs:
        for ident in p.get('identifiers', []) or []:
            # Use (source, id) tuple for uniqueness
            unique_identifiers.add((ident.get('source'), ident.get('id')))

    return {
        'meta': {
            'count': len(profs),
            'last_update': last_update,
            'total_infoscience_pubs': total_infoscience,
            'total_openalex_pubs': total_openalex,
            'total_pubs': total_pubs,
            'total_unique_identifiers': len(unique_identifiers),
        },
        'results': profs
    }


def load_all_profs(include_centres: bool = False) -> tuple[list[dict], str]:
    """
    Load all professors from the database.

    Args:
        include_centres: If False (default), filter out labs with unit_type='CENTRE'
                        and exclude professors who have no remaining labs.

    Returns:
        Tuple of (list of prof dicts, last_update timestamp)
    """
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
                           SELECT sciper, cf, role, unit_name, unit_type, cf_level_2, cf_level_3, acronym_level_2, acronym_level_3
                           FROM sciper_lab
                                    INNER JOIN lab USING (cf)
                           """).df()
        labs_grouped = (
            labs.groupby('sciper')
            .apply(lambda x: x[['cf', 'role', 'unit_name', 'unit_type', 'cf_level_2', 'cf_level_3', 'acronym_level_2', 'acronym_level_3']].to_dict('records'), include_groups=False)
            .reset_index(name='labs')
        )
        df = df.merge(labs_grouped, on='sciper', how='left')

        # Filter out CENTRE labs if include_centres is False
        if not include_centres:
            df['labs'] = df['labs'].apply(
                lambda labs_list: [lab for lab in labs_list if lab.get('unit_type') != 'CENTRE']
                if isinstance(labs_list, list) else []
            )

        df['n_labs'] = df['labs'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        # Remove professors with no labs after filtering
        df = df[df['n_labs'] > 0]

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
                SELECT DISTINCT sciper, name \
                FROM (SELECT sciper, lastname AS name \
                      FROM prof \
                      UNION \
                      SELECT sciper, CONCAT(firstname, ' ', lastname) AS name \
                      FROM prof \
                      UNION \
                      SELECT sciper, CONCAT(lastname, ' ', firstname) AS name \
                      FROM prof \
                      UNION \
                      SELECT sciper, CONCAT(lastname, ' ', firstname) AS name \
                      FROM orcid \
                               INNER JOIN sciper_orcid_integration USING (orcid) \
                      UNION \
                      SELECT sciper, CONCAT(firstname, ' ', lastname) AS name \
                      FROM orcid \
                               INNER JOIN sciper_orcid_integration USING (orcid) \
                      UNION \
                      SELECT sciper, CONCAT(firstname, ' ', lastname) AS name \
                      FROM orcid \
                               INNER JOIN sciper_openalex USING (orcid) \
                      UNION \
                      SELECT sciper, CONCAT(lastname, ' ', firstname) AS name \
                      FROM orcid \
                               INNER JOIN sciper_openalex USING (orcid) \
                      UNION \
                      SELECT sciper, display_name AS name \
                      FROM sciper_openalex \
                      UNION \
                      SELECT sciper, n AS name \
                      FROM (SELECT sciper, UNNEST(other_names) AS n \
                            FROM orcid \
                                     INNER JOIN sciper_openalex USING (orcid) \
                            WHERE other_names IS NOT NULL) \
                      UNION \
                      SELECT sciper, n AS name \
                      FROM (SELECT sciper, UNNEST(other_names) AS n \
                            FROM orcid \
                                     INNER JOIN sciper_orcid_integration USING (orcid) \
                            WHERE other_names IS NOT NULL) \
                      UNION \
                      SELECT sciper, n AS name \
                      FROM (SELECT sciper, UNNEST(display_name_alternatives) AS n \
                            FROM sciper_openalex \
                            WHERE display_name_alternatives IS NOT NULL)); \
                '''
        data = con.execute(query).df()
        data['name'] = data['name'].str.lower()
        data['name'] = data['name'].apply(normalize_name)
        data = data.drop_duplicates()
        names = data.groupby('sciper')['name'].agg(list)

        df = df.merge(names, on='sciper', how='left').rename({'name': 'alternative_names'}, axis=1)
        df = df.where(pd.notnull(df), None)

        # Get last update timestamp
        last_update = con.execute("SELECT timestamp FROM metadata").df()['timestamp'].tolist()[0]

        return df.to_dict(orient='records'), last_update

    finally:
        con.close()


def get_representatives():
    """Load all representative embeddings from the database."""
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
                     SELECT sciper, type, weight, embedding
                     FROM representative
                     """).df()
    con.close()
    return df


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/profs", summary="List all EPFL professors")
def get_profs(
        include_centres: bool = Query(default=False, description="Include labs with unit_type='CENTRE'. If False, professors with only CENTRE labs are excluded.")
):
    """Get list of all professors with their labs, publications, and identifiers extracted from Infoscience, OpenAlex, and Orcid"""
    profs, last_update = load_all_profs(include_centres=include_centres)
    return build_response(profs, last_update)


@app.get("/profs/{sciper}", summary="Retrieve specific EPFL professor")
def get_prof(
        sciper: int = PathParam(..., examples=[105782],
                                description="SCIPER of the professor (e.g., you can try with 105782)"),
        include_centres: bool = Query(default=False, description="Include labs with unit_type='CENTRE'. If False, professors with only CENTRE labs are excluded.")
):
    """Get a single person by sciper."""
    profs, last_update = load_all_profs(include_centres=include_centres)

    for p in profs:
        if p['sciper'] == sciper:
            return build_response([p], last_update)

    raise HTTPException(status_code=404, detail="Person not found")


@app.get("/profs/name/{fullname}", summary="Search a professor by their fullname using a fuzzy match")
def search_prof(
        fullname: str = PathParam(..., examples=['Pascal Frossard'], description="Name of the professor."),
        limit: int = Query(default=5, ge=1, le=100, description="Maximum number of results to return"),
        include_centres: bool = Query(default=False, description="Include labs with unit_type='CENTRE'. If False, professors with only CENTRE labs are excluded.")
):
    """Search professors by name using fuzzy matching. Returns results sorted by relevance."""
    profs, last_update = load_all_profs(include_centres=include_centres)
    profs_by_sciper = {x['sciper']: x for x in profs}

    query = normalize_name(fullname)
    scores = []

    for sciper, prof in profs_by_sciper.items():
        alt_names = prof.get('alternative_names') or []
        if not alt_names:
            continue
        best_score = max(
            fuzz.partial_ratio(query, normalize_name(name))
            for name in alt_names
        )
        scores.append({'sciper': sciper, 'score': best_score})

    if not scores:
        raise HTTPException(status_code=404, detail="Not found")

    scores.sort(key=lambda x: x['score'], reverse=True)

    output = []
    for item in scores[:limit]:
        prof = profs_by_sciper[item['sciper']].copy()
        prof['match_score'] = item['score']
        output.append(prof)

    return build_response(output, last_update)


@app.get("/profs/cf/{cf}", summary="Get all professors from a lab by its CF identifier")
def search_prof_by_cf(
        cf: int = PathParam(..., examples=[929], description="CF identifier of the lab."),
        limit: int = Query(default=None, ge=1, le=1000,
                           description="Maximum number of results to return (default: all)"),
        include_centres: bool = Query(default=False, description="Include labs with unit_type='CENTRE'. If False, professors with only CENTRE labs are excluded.")
):
    """Get all professors belonging to a specific lab."""
    profs, last_update = load_all_profs(include_centres=include_centres)

    output = []
    for prof in profs:
        for lab in prof.get('labs') or []:
            if lab['cf'] == cf:
                output.append(prof)
                break  # avoid duplicates if prof has same cf multiple times

    if not output:
        raise HTTPException(status_code=404, detail=f"No professors found for CF {cf}")

    if limit is not None:
        output = output[:limit]

    return build_response(output, last_update)


@app.get("/profs/semantic/{topic}", summary="Search professors by research topic using semantic similarity")
def search_prof_by_topic(
        topic: str = PathParam(..., examples=['machine learning', 'network security', 'quantum computing'],
                               description="Research topic to search"),
        method: SearchMethod = Query(default=SearchMethod.mixed,
                                     description="Search method: mean, recency, cluster, or mixed (weighted average of all three)"),
        limit: int = Query(default=5, ge=1, le=100, description="Maximum number of results to return"),
        include_centres: bool = Query(default=False, description="Include labs with unit_type='CENTRE'. If False, professors with only CENTRE labs are excluded.")
):
    """Find professors whose research is most similar to the given topic."""
    profs, last_update = load_all_profs(include_centres=include_centres)
    profs_by_sciper = {x['sciper']: x for x in profs}

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
    for item in final_scores[:limit]:
        prof = profs_by_sciper[item['sciper']].copy()
        prof['match_score'] = round(item['score'], 4)
        output.append(prof)

    return build_response(output, last_update)


if __name__ == "__main__":
    result = search_prof_by_topic('machine learning', method=SearchMethod.mixed)
    print(result)