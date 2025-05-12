"""
Download.py  –  Save the HTML of a single URL.

Usage:
    python Download.py https://example.com
"""

import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from collections import deque
from urllib.parse import urlparse, urljoin


def download(url: str) -> str:
    """Return the HTML of *url* as a Unicode string."""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()          # bubble up HTTP errors
    return resp.text


def default_filename(url: str) -> Path:
    """Save into <netloc>/<everything-after-netloc>, falling back to index.html."""
    p = urlparse(url)
    root = Path(p.netloc)
    sub = p.path.lstrip("/") + ".html"
    target = root / sub
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def main():
    if len(sys.argv) < 2:
        print("Usage: python Download.py <url>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    base_host = urlparse(url).netloc
    url_queue = deque([url])
    visited = set()
    
    while url_queue:
        url = url_queue.popleft()

        try:
            html = download(url)
            visited.add(url)
        except Exception as exc:
            print(f"Couldn’t fetch {url}: {exc}", file=sys.stderr)
            continue

        clean_url = url.split('#',1)[0].split('?',1)[0] #Removes characters windows does not allow in filenames
        outfile = default_filename(clean_url)
        outfile.write_text(html, encoding="utf-8")
        print(f"Saved {url} → {outfile.resolve()}")

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all("a", href=True):
            link = urljoin(url, tag["href"])     # resolve relative → absolute
            if urlparse(link).netloc.endswith(base_host) and (link not in visited) and (link not in url_queue):
                url_queue.append(link)

        if len(visited) >= 50: break

if __name__ == "__main__":
    main()
