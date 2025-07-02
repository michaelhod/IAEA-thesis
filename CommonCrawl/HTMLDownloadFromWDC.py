#!/usr/bin/env python3
"""
This script samples random URLs from the Web Data Commons. These are URLs with schema.org.
It downloads the raw HTML of the pages and saves them to a local directory.
"""

import random, re, gzip, pathlib, urllib.parse, requests
from bs4 import BeautifulSoup      # pip install beautifulsoup4

# -------------------------------------------------------------------
# CONFIGURATION
BASE_DIRS = [
    "LocalBusiness",
    "GovernmentOrganization",
    "Movie"
]
BASE_URL  = ("https://data.dws.informatik.uni-mannheim.de/"
             "structureddata/2024-12/quads/classspecific")
UA        = "wdc-lazy-sampler/0.1 (+you@example.com)"
SEED      = 42
MAX_SKIP  = 50_000                 # how many lines to skip max
OUT_DIR   = pathlib.Path("wdc_lazy_html")
OUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)

# -------------------------------------------------------------------
def list_chunks(subset: str) -> list[str]:
    """Scrape the directory listing to get every *.nq.gz file name."""
    url = f"{BASE_URL}/{subset}/"
    res = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    return [
        a["href"] for a in soup.select("a[href$='.nq.gz']") if a["href"].endswith(".nq.gz")
    ]

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
                    continue                      # keep skipping
                # line is bytes → decode
                parts = line.decode("utf-8", "ignore").rsplit(" ", 2)
                if len(parts) < 2:
                    continue
                ctx = parts[-2].lstrip("<").rstrip("> .")
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
    # 1. choose a subset and a random chunk
    subset = random.choice(BASE_DIRS)
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
    out  = OUT_DIR / f"{urllib.parse.quote_plus(url)}.html"
    out.write_text(html, encoding="utf-8", errors="ignore")
    print("Saved HTML →", out.resolve())

if __name__ == "__main__":
    main()
