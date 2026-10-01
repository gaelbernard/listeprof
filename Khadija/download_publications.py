import json
from pathlib import Path

from dotenv import load_dotenv
from dspace_rest_client.client import DSpaceClient


def download_publications(sciper, year_min, year_max):
    infoscience = DSpaceClient()
    infoscience.authenticate()

    query = f"dc.date.issued:[{year_min}-01-01 TO {year_max}-12-31] AND dspace.entity.type:publication"
    dsos = infoscience.search_objects(query=query, configuration="researchoutputs")

    publications = []
    for item in dsos:
        data = item.as_dict()
        metadata = data.get("metadata", {})
        authors_sciper = {
            int(s["value"])
            for s in metadata.get("cris.virtual.sciperId") or []
            if s.get("value") and s["value"].isdigit()
        }
        if sciper not in authors_sciper:
            continue
        publications.append(data)

    return publications


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")

    sciper = 105074
    year_min = 2018
    year_max = 2025

    publications = download_publications(sciper, year_min, year_max)

    output = root / f"publications_{sciper}.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(publications, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(publications)} publications to {output}")
