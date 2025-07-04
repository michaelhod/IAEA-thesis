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
import csv

PRINT_URLS = False

#Array of domains visited, imported from csv
DOMAINS_VISITED = []
with open("domains_visited.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        DOMAINS_VISITED.append(row[0])

WDC_Subsets = [
    "AdministrativeArea",
    "Answer",
    "CollegeOrUniversity",
    "Country",
    "EducationalOrganization",
    "LocalBusiness",
    "Organization",
    "GovernmentOrganization",
    "Person",
    "Place"
]
NUM_SAMPLES = [ # how many web pages to take
    0, # AdministrativeArea
    0, # Answer
    2, # CollegeOrUniversity
    0, # Country
    2, # EducationalOrganization
    1, # LocalBusiness
    2, # Organization
    2, # GovernmentOrganization
    0, # Person
    2  # Place
]
BASE_URL = ("https://data.dws.informatik.uni-mannheim.de/"
             "structureddata/2024-12/quads/classspecific")
UA = 'cc-schemaxtract/1.0 (JSON-LD & microdata extractor; michaelhodgins@live.co.uk)'
SEED = None
MAX_SKIP = 1_000_000 # The max possible number of lines to skip in .gz files
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

def random_url_from_gz_file(subset: str, gz_file: str, num_urls: int) -> str | None:
    """Open the gz file, skip random(1..MAX_SKIP) lines, return next URL."""
    extract_at = sorted(random.sample(range(1, MAX_SKIP + 1), num_urls))
    gz_url = f"{BASE_URL}/{subset}/{gz_file}"
    urls = []
    domains = []  # To avoid duplicates
    with requests.get(gz_url, headers={"User-Agent": UA},
                      stream=True, timeout=60) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gzf:
            for i, line in enumerate(gzf, start=1):
                if i > MAX_SKIP:
                    break
                
                if i not in extract_at:
                    continue

                parts = line.decode("utf-8", "ignore").split(" ")

                if len(parts) < 4:
                    extract_at[extract_at.index(i)] += 1 # Try the next line
                    continue

                ctx = parts[3].lstrip("<").rstrip("> .")

                if not ctx.startswith("http"):
                    extract_at[extract_at.index(i)] += 1 # Try the next line
                    continue
                
                # Avoid duplicates by checking the domain
                domain = urllib.parse.urlparse(ctx).netloc
                if domain not in domains and domain not in DOMAINS_VISITED:
                    domains.append(domain)
                    urls.append(ctx)

            return urls

def fetch_html(url: str) -> str:
    try:
        resp = requests.get(url, headers={"User-Agent": UA},timeout=5)
        resp.raise_for_status()
        return resp.text
    
    except requests.exceptions.RequestException as e:
        if PRINT_URLS:
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
    for (subset, num_samples) in zip(WDC_Subsets, NUM_SAMPLES):
        if num_samples <= 0:
            continue

        # 1. choose a random gz_file
        gz_files = list_gz_files(subset)
        if not gz_files:
            raise RuntimeError(f"No gz_files found for {subset}")
        gz_file  = random.choice(gz_files)
        print()
        print(f"Subset = {subset}  |  gz_file = {gz_file} |  num_samples = {num_samples}")
        
        # 2. pick a random URL
        urls = random_url_from_gz_file(subset, gz_file, num_samples*100) # *10 creates a buffer incase websites are not usable  # Shuffle to randomize the order
        if len(urls) == 0:
            raise RuntimeError("Couldn’t find a usable URL – try again")
        random.shuffle(urls)

        print("Downloading HTML")
        num_samples_saved = 0
        for url in urls:
            url_variations = [url, "https://" + urllib.parse.urlparse(url).netloc]
            for url_variation in url_variations:
                if PRINT_URLS:
                    print()
                # 3. download raw HTML
                if PRINT_URLS:
                    print("Chosen URL:", url_variation)

                html = fetch_html(url_variation)

                if not html:
                    if PRINT_URLS:
                        print("No HTML")
                    continue

                if "itemscope" not in html:
                    if PRINT_URLS:
                        print("No Microdata")
                    continue

                if not is_english(html):
                    if PRINT_URLS:
                        print("Not English")
                    continue

                # 4. save HTML to disk
                out  = OUT_DIR / subset / f"{urllib.parse.quote_plus(url_variation)}.html"
                out.write_text(html, encoding="utf-8", errors="ignore")

                with open("domains_visited.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([urllib.parse.urlparse(url_variation).netloc])

                if PRINT_URLS:
                    print("Saved HTML →", out.resolve())

                num_samples_saved += 1
                print(num_samples_saved, end=", ")
            
            if num_samples_saved >= num_samples*len(url_variations):
                break

if __name__ == "__main__":
    main()
