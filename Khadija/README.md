# Download publications from Infoscience

Minimal example that downloads publications for one EPFL researcher (by **SCIPER**) and saves them as JSON.

## Prerequisites

- Python 3.9 or newer
- An EPFL account with access to [Infoscience](https://infoscience.epfl.ch/)
- A personal API token from Infoscience

## 1. Get your Infoscience API token

1. Log in to Infoscience with your EPFL account: [https://infoscience.epfl.ch/profile](https://infoscience.epfl.ch/profile)
2. On your profile page, find **Token d'accès personnel** (personal access token).
3. Copy the token.

## 2. Set up the environment

From this folder (project root):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs the EPFL **DSpace-CRIS** Python client (`dspace-rest-python`) from [github.com/epfllibrary/dspace-rest-python](https://github.com/epfllibrary/dspace-rest-python) (branch `dev`).

## 3. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```bash
DS_API_TOKEN=<paste your token from Infoscience profile>
DS_API_ENDPOINT=https://infoscience.epfl.ch/server/api
```

## 4. Run the script

```bash
python download_publications.py
```

Or call the function directly:

```python
from download_publications import download_publications

publications = download_publications(sciper=105074, year_min=2018, year_max=2025)
```

This creates `publications_<sciper>.json` with the full Infoscience data for each matching publication.

## References

- Infoscience profile (token): [https://infoscience.epfl.ch/profile](https://infoscience.epfl.ch/profile)
- Infoscience API help: [https://help-infoscience.epfl.ch/api.html](https://help-infoscience.epfl.ch/api.html)
- EPFL Python client: [https://github.com/epfllibrary/dspace-rest-python](https://github.com/epfllibrary/dspace-rest-python)

## Needs help?
Julien.Sicot@epfl.ch is probably the best person to help with this. 
