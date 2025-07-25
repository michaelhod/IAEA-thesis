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
SRC_FOLDER1 = Path("./data/swde/sourceCode/sourceCode/movie")
SRC_FOLDER2 = Path("./data/swde/sourceCode/sourceCode/nbaplayer")
SRC_FOLDER3 = Path("./data/swde/sourceCode/sourceCode/university")
OUT_ROOT1   = Path("./data/swde_HTMLgraphs/movie")
OUT_ROOT2   = Path("./data/swde_HTMLgraphs/nbaplayer")
OUT_ROOT3   = Path("./data/swde_HTMLgraphs/university")
OUT_ROOT1.mkdir(parents=True, exist_ok=True)
OUT_ROOT2.mkdir(parents=True, exist_ok=True)
OUT_ROOT3.mkdir(parents=True, exist_ok=True)
# SRC_FOLDER_WDC = Path("./data/wdc_microdata_html")
# OUT_ROOT_WDC = Path("./data/wdc_microdata_HTMLgraphs")
# OUT_ROOT_WDC.mkdir(parents=True, exist_ok=True)

# ── worker ──────────────────────────────────────────────────────────────────────
def process_file(filepath: Path, SRC: Path, OUT: Path) -> str | None:
    # build a parallel directory structure under OUT_ROOT
    rel   = filepath.relative_to(SRC)
    out_dir = (OUT / rel).with_suffix("")

    if (out_dir/"A.npz").exists():
        return f"{out_dir} already written"

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        A, X, E, edge_index, bbox = html_to_graph(filepath, get_Driver(), OverwriteHTML=False)
        
        # save arrays
        A = sparse.csr_matrix(A)
        X = sparse.csr_matrix(X)
        E = sparse.csr_matrix(E)

        sparse.save_npz(out_dir / "A.npz", A, compressed=True)
        sparse.save_npz(out_dir / "X.npz", X, compressed=True)
        sparse.save_npz(out_dir / "E.npz", E, compressed=True)
        np.save(out_dir / "edge_index.npy", edge_index)
        pd.DataFrame(bbox).to_csv("bbox.csv", index=False)


        return f"Saved: {str(out_dir)}"

    # except NoSuchElementException as e:
    #     print("Retrying")
    #     try:
    #         A, X, E, edge_index = html_to_graph(filepath, get_Driver(), OverwriteHTML=True)
            
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
    #         pd.DataFrame(bbox).to_csv("bbox.csv", index=False)
    
    #         return f"Saved: {str(out_dir)}"
        
    #     except Exception as e:
    #         print(f"{filepath.absolute().resolve()}: {e}")
    #         return None

    except Exception as e:
        with open('./data/skipped.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(filepath)
        print(f"{filepath.absolute().resolve()}: {e}")
        return None

# ── main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for src, out in zip([SRC_FOLDER3, SRC_FOLDER1, SRC_FOLDER2],[OUT_ROOT3, OUT_ROOT1, OUT_ROOT2]):
        html_files = list(src.rglob("*.htm"))
        batchsize = len(html_files)
        workers = None
        for i in range(0, len(html_files), batchsize):
            batch = html_files[i:i+batchsize]
            with ProcessPoolExecutor(max_workers=workers, initializer=driver_init, initargs=(True,)) as pool:
                for saved_to in pool.map(process_file, batch, repeat(src, len(batch)), repeat(out, len(batch)), chunksize=1):
                    if saved_to:
                        print(saved_to)
                    else:
                        print("Error, skipping and restarting Chrome...")
                        restart_Driver(False)
                        # print("Error, restarting pool and Chrome processes...")
                        # pool.shutdown(wait=True, cancel_futures=True)
                        # break
            print("Restarting Selenium drivers...")
            time.sleep(1)
            subprocess.run(r'taskkill /f /im chrome.exe /im chromedriver.exe', shell=True)