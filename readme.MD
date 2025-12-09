



# ProfDB – EPFL Professor Data Pipeline

## Overview

ProfDB is a data pipeline that takes an official list of EPFL professors (from a CSV export) and builds a consolidated DuckDB database with enriched information. The pipeline augments the professor list with their recent publications (from EPFL's Infoscience repository and OpenAlex) and researcher identifiers (OpenAlex author IDs and ORCIDs). The resulting DuckDB database can be served via a FastAPI service for convenient querying of professors and their publications. The goal is to make the data easily accessible through an API while keeping it up-to-date with minimal manual intervention.

## Input Data

The pipeline starts with an official EPFL professor list CSV. This CSV is downloaded from EPFL's InfoView system and contains one row per professor with columns such as: `N° sciper` (unique ID), `email`, `Nom` (last name), `Prénom` (first name), `date création` (creation date), `classe acc.` (academic rank), `CF unité` (unit code), `Nom unité` (unit name), `fonction acc.` (role). These columns are required for the pipeline to run; the code will validate that they exist. Make sure your input CSV has these headers (in French, as in the official source).

## Pipeline Steps

### 1. Load Professors and Labs (CSV Ingestion)

The pipeline reads the input CSV into a pandas DataFrame. It then creates three core tables in DuckDB:

- **prof** – a table of professors with their SCIPER ID, email, first name, last name, creation date, and academic class. Duplicate entries are dropped and entries are sorted by name.
- **lab** – a table of unique labs/units, identified by the unit code (`CF unité`) and unit name.
- **sciper_lab** – a linking table associating each professor (sciper) with their unit(s) and role(s).

These tables are written into the DuckDB database, with indexes on key fields (sciper, unit code) for performance.

### 2. ORCID Integration Lookup

Next, the pipeline fetches ORCID identifiers from EPFL's internal ORCID Integration service. It calls the ORCID integration API (an EPFL endpoint) to retrieve a JSON mapping of SCIPER IDs to ORCID iDs. The result is filtered to include only the ORCIDs for professors present in our `prof` table. This produces a table `sciper_orcid_integration` with columns `sciper` and `orcid` (one row per professor who has an ORCID in the integration system). An index on the `sciper` field is created for fast lookups.

### 3. Import Publications from Infoscience

The pipeline then connects to Infoscience (EPFL's institutional repository on DSpace) to gather recent publications for each professor. Using the provided year range (e.g., 2025–2025 for only the year 2025), it queries Infoscience for all publications in that date range. For each publication found, it checks if any of the authors have a SCIPER ID matching one of our professors. If none of the authors are in our professor list, that publication is skipped.

All qualifying publications are collected and assigned a new local publication ID (`id_pub`). Key metadata is stored: Infoscience ID/URL, DOI, title, abstract, and the publication date. At the same time, a `sciper_pub` association table is built to link each professor (sciper) to the publications (`id_pub`) they authored.

After iterating through the results, the code creates a `pub` table in DuckDB with all collected publications and their metadata (including a computed `year_issued`), and a `sciper_pub` table linking professors to those publications. Indexes are added on publication IDs and sciper IDs for efficient joins.

### 4. Identify OpenAlex Author IDs

The pipeline then tries to find each professor's OpenAlex author ID using an algorithmic approach. For each professor, it takes up to 50 of their DOIs from the publications gathered in the previous step (preferably the most recent ones). It queries the OpenAlex API for these DOIs, retrieving the list of authors for each of those works. This yields a pool of candidate author profiles from OpenAlex.

The pipeline counts how frequently each OpenAlex author appears across the professor's DOI set and then compares the author names to the professor's name. It standardizes names (to handle initials, accents, etc.) and computes a Levenshtein edit distance between the professor's name and each candidate author name. If it finds one or more candidates with a very close name match (edit distance ≤ 1), those OpenAlex IDs are deemed to correspond to the professor. (If no close match is found, it currently defaults to none, though in principle one could take the top candidates.)

This step produces a `sciper_openalex` table mapping each professor (sciper) to their inferred OpenAlex author ID(s), along with the author's details from OpenAlex. (The OpenAlex author details include fields like display name and may include ORCID if available.) A couple of known mismatches can be corrected via a manual override list in the code (for specific professors). Indexes on sciper and OpenAlex ID are added to this table.

### 5. Augment with OpenAlex Publications

Using the OpenAlex author IDs from the previous step, the pipeline next retrieves additional publications for each professor from OpenAlex (to include publications that might not be in Infoscience). For each professor-author ID, it calls the OpenAlex Works API to get all works authored by that ID in the given year range, filtering to certain publication types (articles, books, book-chapters). The OpenAlex API is queried in pages (cursor-based pagination) to retrieve up to all relevant works.

For each OpenAlex work found, the pipeline checks if that publication was already in our `pub` table (from Infoscience) by comparing DOI or a fuzzy match on title (using Levenshtein distance < 3):

- If a publication already exists (meaning it was in Infoscience data), the code updates that existing record: it fills in the OpenAlex work ID and adds the abstract if our record lacked one.
- If the publication is new (not in Infoscience), it is added as a new entry to the `pub` table with a fresh `id_pub`, marking `id_infoscience` as `NULL` (since it's not in the internal repository). The new entry includes the DOI, title, abstract (reconstructed from OpenAlex's data), publication year, and OpenAlex work ID.

(Under the hood, the OpenAlex response provides an abstract inversion index, which the code reverses to construct the abstract text, and the titles are cleaned of special characters.)

> **Note:** Each new OpenAlex-sourced publication should also be linked to the professor in the `sciper_pub` table – since these were fetched per professor, we infer that these works belong to that professor's record. (The code ensures no duplicates and handles cases where Infoscience and OpenAlex overlap for the same work.)

### 6. Fetch ORCID Profiles

In this step, the pipeline consolidates ORCID information for the professors. It takes the union of ORCID iDs found via the internal integration (step 2) and those that were found via OpenAlex author profiles. This yields a set of unique ORCID identifiers for our professors.

It then calls the public ORCID API for each ORCID iD (using the public ORCID v3.0 API to retrieve the full profile in JSON). From each ORCID profile, the code extracts various details: whether the profile is claimed by the researcher, the full name (given and family names), other names (aliases), email addresses, biography, and keywords. It also collects the researcher's web URLs and external identifiers (like ResearcherID, Scopus ID, etc.) from the ORCID record.

All these details are stored in two tables:

- **orcid** – one row per ORCID with personal info and flags
- **orcid_links** – multiple rows per ORCID, listing each external link or identifier associated with that ORCID

This provides enriched metadata for professors who have ORCID profiles.

### 7. Compute Publication Embeddings

Finally, the pipeline generates text embeddings for each publication's title and abstract. This is useful for search or similarity queries via the API. The code uses OpenAI's embedding API (with a chosen model, e.g., `text-embedding-3-small` or similar) to compute a numeric vector for each publication's concatenated title+abstract.

To optimize this process, it employs a local LMDB cache: for each unique text, it first checks if an embedding was already computed in a previous run (and saved in the cache) to avoid re-computation. It processes publications in batches (configurable batch sizes, e.g., 64 per API call).

Embeddings are retrieved from OpenAI in batches and stored as numpy arrays, then inserted into the DuckDB as a `pub_embedding` table (with each row having `id_pub` and an embedding vector). The embedding table is freshly created on each run. At the end of this step, every publication in the `pub` table has a corresponding vector in `pub_embedding` (either newly computed or from cache), enabling semantic searches through the FastAPI if needed. The log will report how many embeddings were computed vs. fetched from cache for transparency.

### 8. Packaging Output

After all the above steps are successfully completed, the pipeline finalizes the output. The DuckDB database (initially built as a temporary file `_temp.duckdb`) is closed and moved to the output directory. Specifically, the pipeline creates an `output/latest/` folder (creating the `output/` directory if not present).

It copies the input CSV into this folder (renaming it to `input_prof.csv` for reference), moves the DuckDB file there as `db.duckdb`, and also saves the log file as `db.log`.

It then generates a convenient Parquet file of all publications with embeddings: it joins the `pub`, `sciper_pub`, and `pub_embedding` tables to get each publication's basic info (title, abstract, id), its source (whether it came from Infoscience or OpenAlex), and the embedding vector, along with the associated professor's sciper. This is written to `output/latest/publications.parquet`.

Finally, to keep an archive of past runs, the entire `latest` folder is copied to a new folder named with a timestamp (e.g., `output/db_20251208_013000/` etc.). In this way, the most recent output is always in `output/latest`, and previous snapshots are retained (timestamped by run date and time). The pipeline also writes a small metadata table inside the DuckDB with the timestamp and total processing time of the run, for auditing purposes.

## Usage and Reusability

### Running the Pipeline

You can run the pipeline by instantiating the `ProfDB` class and calling its `build()` method. For example, in Python:

```python
from code.profdb import ProfDB

csv_path = "input/professors_list.csv"  # path to the EPFL professor list CSV
profdb = ProfDB(csv_path, year_min=2025, year_max=2025)
profdb.build()
```

This will execute all the steps described above, using the CSV as input and the year range 2025–2025 for publications (you can adjust the years as needed). After running, check the `output/latest/` directory for the results.

### Requirements

Ensure you have network access to the required services:

- **Infoscience DSpace API:** The code uses an EPFL DSpace REST client to query publications. You might need appropriate network access or credentials (the code calls `DSpaceClient().authenticate()` by default, which for EPFL internal usage should suffice).

- **OpenAlex API:** No special credentials are required, but an OpenAlex API key is recommended to increase request limits. The pipeline will read an `API_KEY_OPENALEX` from your environment (e.g., a `.env` file) if provided. Make sure to set this environment variable with your OpenAlex key (or add it to a `.env` file in the working directory).

- **ORCID API:** The ORCID public API is used (no auth token needed for public data). The code simply makes GET requests to `pub.orcid.org` for each ORCID ID.

- **OpenAI API:** An OpenAI API key is required for the embeddings step. Set the `OPENAI_API_KEY` environment variable before running, so that the OpenAI Python client can authenticate. If you don't have this or don't want to generate embeddings, you could disable or remove the embedding step, but by default it will try to run it.

You need a .env file at the root with the following info:
```bash
DS_API_TOKEN=eyJhbGc...
DS_API_ENDPOINT=https://infoscience.epfl.ch/server/api
OPENAI_API_KEY=sk-proj-qX2...
API_KEY_OPENALEX=wUpK...
```
The tool should work too without the API_KEY_OPENALEX (nice-to-have)

### Docker Usage

To facilitate reusability and scheduling, you can run this pipeline in a Docker container. Build the Docker image from the repository root (assuming the code and a proper Dockerfile are present) with a command such as:

```bash
docker build -t profdb-pipeline .
```

This will create an image with the pipeline. You can then run the container, mounting the input and output directories as needed. For example:

```bash
docker run --rm -v /path/to/input:/app/input -v /path/to/output:/app/output profdb-pipeline
```

This assumes the Dockerfile is set up to copy the code and run the `ProfDB.build()` on container start. The above command mounts your local input folder (containing the CSV) and an output folder to persist results outside the container. Adjust paths and image name as appropriate.

The pipeline is designed to run to completion and then exit, so it fits well with a batch job or cron schedule. (In our case, we schedule it to run every week on Sunday night to refresh the data automatically.)

### FastAPI Integration

The API is self-documented here: 
[http://itsisa0052.xaas.epfl.ch:5001/docs](http://itsisa0052.xaas.epfl.ch:5001/docs)


## Output Structure

After a successful run, your `output/latest` directory will contain:

- **db.duckdb** – the DuckDB database file with all the tables (`prof`, `lab`, `sciper_lab`, `sciper_orcid_integration`, `pub`, `sciper_pub`, `sciper_openalex`, `orcid`, `orcid_links`, `pub_embedding`, and `metadata`). You can open this with DuckDB or any compatible tool to explore the data.

- **db.log** – the log file capturing the pipeline execution details (useful for debugging or verifying that each step completed).

- **input_prof.csv** – a copy of the input CSV that was used, for record-keeping.

- **publications.parquet** – a Parquet file containing a flattened view of all professors' publications with selected fields. Each row in this file includes the professor's `sciper`, publication `title`, `abstract`, internal `id_pub`, the `source` of the record ("infoscience" or "openalex"), and the `embedding` vector. This file can be handy for quickly loading into analytics tools or for powering search functionality in the API.

Additionally, the pipeline creates a timestamped archive on each run. For example, you might see folders like `output/db_20251208_013000/` (with a datetime stamp). These are copies of what was in `latest` at the moment of that run's completion. This versioning allows you to go back to previous snapshots if needed, while always having `output/latest` point to the most current data.


