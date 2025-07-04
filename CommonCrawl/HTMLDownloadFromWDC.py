#!/usr/bin/env python3
"""
This script samples random URLs from the Web Data Commons. These are URLs with schema.org.
It downloads the raw HTML of the pages and saves them to a local directory.

1) Chose a random subset from the Web Data Commons schema.org subsets.
2) Scrape the subset to get every *.gz file name.
3) Pick a random *.gz file, and pick a random URL from it.
"""

import random, re, gzip, pathlib, urllib.parse, requests
from bs4 import BeautifulSoup

WDC_Subsets = [
    "LocalBusiness",
    "GovernmentOrganization",
    "Movie"
]
BASE_URL = ("https://data.dws.informatik.uni-mannheim.de/"
             "structureddata/2024-12/quads/classspecific")
UA = 'cc-schemaxtract/1.0 (JSON-LD & microdata extractor; michaelhodgins@live.co.uk)'
NUM_SAMPLES = 30 # how many web pages to take
SEED = None
MAX_SKIP = 100_000 # The max possible number of lines to skip in .gz files
OUT_DIR = pathlib.Path("./wdc_microdata_html")
OUT_DIR.mkdir(exist_ok=True)
for subset in WDC_Subsets:
    (OUT_DIR / subset).mkdir(exist_ok=True)

random.seed(SEED)

# -------------------------------------------------------------------
def list_gz_files(subset: str) -> list[str]:
    """Get every *.gz file name for a WDC_SUBSET."""
    url = f"{BASE_URL}/{subset}/"
    res = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    return [a["href"] for a in soup.select("a[href$='.gz']")]

def random_url_from_gz_file(subset: str, gz_file: str) -> str | None:
    """Open the gz file, skip random(1..MAX_SKIP) lines, return next URL."""
    start_line = random.randint(1, MAX_SKIP)
    gz_url = f"{BASE_URL}/{subset}/{gz_file}"
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
    try:
        resp = requests.get(url, headers={"User-Agent": UA},timeout=30)
        resp.raise_for_status()
        return resp.text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def is_english(html: str) -> bool:
    """Check if the HTML content is in English."""
    soup = BeautifulSoup(html, "html.parser")

    lang_attr = soup.html.get("lang")
    if lang_attr:
        if lang_attr.split("-")[0].lower() == "en":
            return True
        return False

    meta_lang = soup.find("meta", attrs={"http-equiv": "content-language"})
    if meta_lang and "en" in (meta_lang.get("content") or "").lower():
        return True
    
    return False


# -------------------------------------------------------------------
def main():
    i = 0
    while i < NUM_SAMPLES:
        # 1. choose a random subset and a random gz_file
        subset = random.choice(WDC_Subsets)
        gz_files = list_gz_files(subset)
        if not gz_files:
            raise RuntimeError(f"No gz_files found for {subset}")
        gz_file  = random.choice(gz_files)
        print()
        print(f"Subset = {subset}  |  gz_file = {gz_file}")
        
        # 2. pick a random URL
        url = random_url_from_gz_file(subset, gz_file)
        if not url:
            raise RuntimeError("Couldn’t find a usable URL – try again")
        print("Chosen URL:", url)

        # 3. download raw HTML
        html = fetch_html(url)

        if not html:
            print("No HTML")
            continue

        if "itemscope" not in html:
            print("No Mircodata")
            continue

        if not is_english(html):
            print("Not English")
            continue

        # 4. save HTML to disk
        out  = OUT_DIR / subset / f"{urllib.parse.quote_plus(url)}.html"
        out.write_text(html, encoding="utf-8", errors="ignore")
        print("Saved HTML →", out.resolve())

        i += 1

if __name__ == "__main__":
    main()
