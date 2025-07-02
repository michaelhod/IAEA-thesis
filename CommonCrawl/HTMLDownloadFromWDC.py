#!/usr/bin/env python3
"""
This script samples random URLs from the Web Data Commons. These are URLs with schema.org.
It downloads the raw HTML of the pages and saves them to a local directory.

1) Chose a random subset from the Web Data Commons schema.org subsets.
2) Scrape the subset to get every *.gz file name.
3) Pick a random *.gz file, and pick a random URL from it.
"""

import random, re, gzip, pathlib, urllib.parse, requests
from bs4 import BeautifulSoup      # pip install beautifulsoup4

# -------------------------------------------------------------------
# CONFIGURATION
WDC_Subsets = [
    "LocalBusiness",
    "GovernmentOrganization",
    "Movie"
]
BASE_URL  = ("https://data.dws.informatik.uni-mannheim.de/"
             "structureddata/2024-12/quads/classspecific")
UA = 'cc-schemaxtract/1.0 (JSON-LD & microdata extractor; michaelhodgins@live.co.uk)'
NUM_SAMPLES = 30              # how many samples to take
SEED      = None
MAX_SKIP  = 100_000                 # how many lines to skip max in .gz files
OUT_DIR   = pathlib.Path("../wdc_microdata_html")
OUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)

# -------------------------------------------------------------------
def list_chunks(subset: str) -> list[str]:
    """Scrape the directory listing to get every *.gz file name."""
    url = f"{BASE_URL}/{subset}/"
    res = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    return [a["href"] for a in soup.select("a[href$='.gz']")]

def random_url_from_chunk(subset: str, chunk: str) -> str | None:
    """Open the gz file, skip random(1..MAX_SKIP) lines, return next URL."""
    start_line = random.randint(1, MAX_SKIP)
    gz_url = f"{BASE_URL}/{subset}/{chunk}"
    with requests.get(gz_url, headers={"User-Agent": UA},
                      stream=True, timeout=60) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gzf:
            for i, line in enumerate(gzf, start=1):
                if i < start_line:
                    continue
                parts = line.decode("utf-8", "ignore").split(" ")
                if len(parts) < 4:
                    continue
                ctx = parts[3].lstrip("<").rstrip("> .")
                if ctx.startswith("http"):
                    return ctx                    # first usable URL
    return None                                   # rare: file exhausted

def fetch_html(url: str) -> str:
    """Try http & https; return HTML text or raise."""
    try:
        return requests.get(url, headers={"User-Agent": UA},
                            timeout=30).text
    except Exception:
        parsed = urllib.parse.urlparse(url)
        alt = parsed._replace(scheme="https" if parsed.scheme=="http" else "http").geturl()
        return requests.get(alt, headers={"User-Agent": UA},
                            timeout=30).text

# -------------------------------------------------------------------
def main():
    i = 0
    while i < NUM_SAMPLES:
        # 1. choose a subset and a random chunk
        subset = random.choice(WDC_Subsets)
        chunks = list_chunks(subset)
        if not chunks:
            raise RuntimeError(f"No chunks found for {subset}")
        chunk  = random.choice(chunks)
        print(f"Subset = {subset}  |  Chunk = {chunk}")

        # 2. pick a random URL
        url = random_url_from_chunk(subset, chunk)
        if not url:
            raise RuntimeError("Couldn’t find a usable URL – try again")
        print("Chosen URL:", url)

        # 3. download raw HTML
        html = fetch_html(url)
        if "itemscope" not in html:
            continue
        out  = OUT_DIR / f"{urllib.parse.quote_plus(url)}.html"
        out.write_text(html, encoding="utf-8", errors="ignore")
        print("Saved HTML →", out.resolve())

        i += 1

if __name__ == "__main__":
    main()
