import re
import pandas as pd
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

def normalize_doi(doi):
    """
    Normalize DOIs by:
      - Returning None for empty/invalid inputs
      - Lowercasing
      - Removing URL and 'doi:' prefixes
      - Stripping spaces and trailing punctuation
    """
    if not doi:
        return None

    d = str(doi).strip().lower()

    # Remove common prefixes
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if d.startswith(prefix):
            d = d[len(prefix):]
            break

    # Remove spaces and trailing punctuation
    d = d.strip().strip(".,;:!?")

    # Simple validity check (basic DOI pattern)
    if not re.match(r"^10\.\d{4,9}/\S+$", d):
        return None

    return d

import hashlib

def text_to_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

