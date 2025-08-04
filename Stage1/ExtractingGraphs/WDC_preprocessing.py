from HTMLtoGraph import html_to_graph
from pathlib import Path
from itertools import repeat
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from seleniumDriver import driver_init, get_Driver, restart_Driver
from scipy import sparse
import subprocess
import time
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import csv

# ── paths ───────────────────────────────────────────────────────────────────────
SRC_FOLDER_WDC = Path("./data/wdc_microdata_html")
OUT_ROOT_WDC = Path("./data/wdc_microdata_HTMLgraphs")
OUT_ROOT_WDC.mkdir(parents=True, exist_ok=True)
JSDISABLED = False
skipped = np.unique(pd.read_csv("./data/skipped.csv", header=None, usecols=[0], encoding="latin-1").to_numpy())
SKIPPEDFILES = [Path(x) for x in skipped]

# ── worker ──────────────────────────────────────────────────────────────────────
def process_file(filepath: Path, SRC: Path, OUT: Path) -> str | None:
    # build a parallel directory structure under OUT_ROOT
    rel   = filepath.relative_to(SRC)
    out_dir = (OUT / rel).with_suffix("")

    if (out_dir/"A.npz").exists():
        return f"{out_dir} already written"
    elif filepath in SKIPPEDFILES:
        return f"{out_dir} already skipped"
    with open(filepath, "rb") as fh:
        raw_bytes = fh.read()
    text = raw_bytes.decode("utf-8", "replace")
    if text.count("itemscope") < 4:
        return f"{out_dir} not enough itemscope"
    if "itemprop" not in text:
        return f"{out_dir} no itemprop"
    try:
        A, X, E, edge_index, bbox = html_to_graph(filepath, get_Driver(), OverwriteHTML=True)
        
        with open('./data/overwritten.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([filepath])

        out_dir.mkdir(parents=True, exist_ok=True)
        # save arrays
        A = sparse.csr_matrix(A)
        X = sparse.csr_matrix(X)
        E = sparse.csr_matrix(E)

        sparse.save_npz(out_dir / "A.npz", A, compressed=True)
        sparse.save_npz(out_dir / "X.npz", X, compressed=True)
        sparse.save_npz(out_dir / "E.npz", E, compressed=True)
        np.save(out_dir / "edge_index.npy", edge_index)
        pd.DataFrame.from_dict(bbox, orient="index").to_csv(out_dir / "bbox.csv")

        return f"Saved: {str(out_dir)}"

    # except NoSuchElementException as e:
    #     print("Retrying")
    #     try:
    #         A, X, E, edge_index = html_to_graph(filepath, get_Driver(), OverwriteHTML=True)
            
    #         out_dir.mkdir(parents=True, exist_ok=True)        
    #         with open('./data/overwritten.csv', 'a') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(filepath)

    #         # save arrays
    #         A = sparse.csr_matrix(A)
    #         X = sparse.csr_matrix(X)
    #         E = sparse.csr_matrix(E)

    #         sparse.save_npz(out_dir / "A.npz", A, compressed=True)
    #         sparse.save_npz(out_dir / "X.npz", X, compressed=True)
    #         sparse.save_npz(out_dir / "E.npz", E, compressed=True)
    #         np.save(out_dir / "edge_index.npy", edge_index)
    #         pd.DataFrame.from_dict(bbox, orient="index").to_csv(out_dir / "bbox.csv")

    #         return f"Saved: {str(out_dir)}"
        
    #     except Exception as e:
    #         print(f"{filepath.absolute().resolve()}: {e}")
    #         restart_Driver(jsDisabled)
    #         return None

    except Exception as e:
        with open('./data/skipped.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([filepath, e])
        #print(f"{filepath.absolute().resolve()}: {e}")
        print(f"{filepath} errored {e.__class__()}")
        restart_Driver(JSDISABLED)
        return None

# ── main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for src, out in zip([SRC_FOLDER_WDC],[OUT_ROOT_WDC]):
        html_files = list(src.rglob("*.html"))
        np.random.shuffle(html_files)
        batchsize = len(html_files)
        workers = 8
        for i in range(0, len(html_files), batchsize):
            batch = html_files[i:i+batchsize]
            with ProcessPoolExecutor(max_workers=workers, initializer=driver_init, initargs=(JSDISABLED,)) as pool:
                for saved_to in pool.map(process_file, batch, repeat(src, len(batch)), repeat(out, len(batch)), chunksize=1):
                    if saved_to:
                        print(saved_to)
                    else:
                        print("Error, skipped and restarted Chrome...")
                        # print("Error, restarting pool and Chrome processes...")
                        # pool.shutdown(wait=True, cancel_futures=True)
                        # break
            print("Restarting Selenium drivers...")
            time.sleep(0.1)
            subprocess.run(r'taskkill /f /im chrome.exe /im chromedriver.exe', shell=True)