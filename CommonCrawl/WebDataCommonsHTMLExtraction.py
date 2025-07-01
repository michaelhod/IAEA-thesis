#!/usr/bin/env python3
"""
Sample two random PLDs (seed-42) from the LocalBusiness,
GovernmentOrganization, or Movie schema.org subsets (WDC 2024-12),
then download the raw HTML of every page that belongs to each PLD.
"""

import random, csv, gzip, os, pathlib, requests, urllib.parse
from collections import defaultdict
from tqdm import tqdm

# ----------------------------------------------------------------------
SEED              = 42
NUM_PLDS_TO_SAMPLE= 2
SUBSETS           = ["LocalBusiness", "GovernmentOrganization", "Movie"]

BASE_URL = "https://data.dws.informatik.uni-mannheim.de/structureddata/2024-12/quads/classspecific/"

UA = 'cc-schemaxtract/1.0 (JSON-LD & microdata extractor; michaelhodgins@live.co.uk)'
OUT_DIR = pathlib.Path("wdc_html")                 # where HTML goes
OUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)

# ----------------------------------------------------------------------
def download(url):
    """Stream-download a (potentially large) file and yield its bytes."""
    with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1 << 16):
            yield chunk

def fetch_small_csv(url, delimiter=","):
    """Return list(rows) for the small helper CSVs."""
    text = requests.get(url, headers={"User-Agent": UA}, timeout=30).text
    return list(csv.reader(text.splitlines(), delimiter=delimiter))

def pick_sample_plds():
    """Load all domain_stats.csv files and sample N unique PLDs."""
    all_plds = []
    for subset in SUBSETS:
        stats_url = f"{BASE_URL}{subset}/{subset}_domain_stats.csv"
        rows = fetch_small_csv(stats_url, delimiter="\t")
       # print(rows[:5])  # Debug: print first 5 rows
        # Each row: PLD, #quads, #URLs, ...
        plds = [(row[0], row[1]) for row in rows]
      #  print()
     #   print(plds[:5])  # Debug: print first 5 PLDs
        all_plds.extend(plds)
    #print(all_plds[:5])  # Debug: print first 5 PLDs from all subsets
    return random.sample(all_plds, NUM_PLDS_TO_SAMPLE)

def build_lookup(subset):
    """Return dict{pld: chunk_name} for one subset."""
    lu_url = f"{BASE_URL}{subset}/{subset}_lookup.csv"
    return {pld: chunk for pld, tld, chunk in fetch_small_csv(lu_url)} # tld is the domain (.com, .de, ...)

def get_chunks_for_plds(plds):
    """
    Map every chosen PLD to the (subset, chunk) it lives in,
    using the lookup files.
    """
    mapping = {}
    for subset in SUBSETS:
        lu = build_lookup(subset)
        for pld in plds:
            if pld in lu:
                mapping[pld] = (subset, lu[pld])
    return mapping

def extract_context_url(nquad_line):
    """Quickly pull the 4th element (context URL) from an N-Quad."""
    parts = nquad_line.rstrip().split(" ", 3)
    return parts[-1].rstrip(" .")

def fetch_html(context_url):
    """Download raw HTML of the page, retrying HTTP→HTTPS if needed."""
    try:
        html = requests.get(context_url, headers={"User-Agent": UA}, timeout=30).text
    except Exception:
        # try flipping scheme:
        parsed = urllib.parse.urlparse(context_url)
        alt = parsed._replace(scheme="https" if parsed.scheme=="http" else "http").geturl()
        html = requests.get(alt, headers={"User-Agent": UA}, timeout=30).text
    return html

# ----------------------------------------------------------------------
def main():
    sampled_plds = pick_sample_plds()
    print("Sampled PLDs:", sampled_plds)

    # Figure out which chunk each PLD sits in
    placements = get_chunks_for_plds(sampled_plds)
    print("PLD placements:", placements)
    if len(placements) != NUM_PLDS_TO_SAMPLE:
        missing = set(sampled_plds) - set(placements)
        raise RuntimeError(f"Couldn’t locate chunks for: {missing}")

    # Iterate chunk-files; pull only N-Quads whose context URL’s domain matches
    pld_to_urls = defaultdict(set)
    for pld, (subset, chunk_name) in placements.items():
        chunk_url = f"{BASE_URL}{subset}/{chunk_name}"
        print(f"→ Streaming chunk {chunk_name} for {pld}")
        gz_stream = download(chunk_url)
        with gzip.GzipFile(fileobj=iter(gz_stream)) as gzf:
            for line in gzf:
                ctx = extract_context_url(line.decode("utf-8", errors="ignore"))
                if ctx.startswith("http") and pld in urllib.parse.urlparse(ctx).netloc:
                    pld_to_urls[pld].add(ctx)

    # Fetch raw HTML for every URL we found
    for pld, urls in pld_to_urls.items():
        print(f"\nDownloading {len(urls)} pages from {pld}")
        abc = 0
        for url in tqdm(urls, desc=pld):
            if abc > 3:
                continue
            abc += 1
            try:
                html = fetch_html(url)
                fn = OUT_DIR / f"{pld}_{urllib.parse.quote_plus(url)}.html"
                fn.write_text(html, encoding="utf-8", errors="ignore")
            except Exception as e:
                tqdm.write(f"⚠️ {url}: {e}")

    print("\nDone! HTML pages are in", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
