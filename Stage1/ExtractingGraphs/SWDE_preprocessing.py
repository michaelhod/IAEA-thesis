from HTMLtoGraph import html_to_graph
from pathlib import Path
from itertools import repeat
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from seleniumDriver import driver_init, get_Driver
from scipy import sparse

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

# ── worker ──────────────────────────────────────────────────────────────────────
def process_file(filepath: Path, SRC: Path, OUT: Path) -> str | None:
    # build a parallel directory structure under OUT_ROOT
    rel   = filepath.relative_to(SRC)
    out_dir = (OUT / rel).with_suffix("")

    if (out_dir/"A.npz").exists():
        return f"{out_dir} already written"

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        html = filepath.read_text(encoding="utf-8")
        A, X, E, edge_index = html_to_graph(html, get_Driver())
        
        # save arrays
        A = sparse.csr_matrix(A)             # convert once
        X = sparse.csr_matrix(X)
        E = sparse.csr_matrix(E)

        sparse.save_npz(out_dir / "A.npz", A, compressed=True)
        sparse.save_npz(out_dir / "X.npz", X, compressed=True)
        sparse.save_npz(out_dir / "E.npz", E, compressed=True)
        np.save(out_dir / "edge_index.npy", edge_index)

        return f"Saved: {str(out_dir)}"

    except Exception as e:
        print(f"{filepath}: {e}")
        return None

# ── main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for src, out in zip([SRC_FOLDER1, SRC_FOLDER2, SRC_FOLDER3],[OUT_ROOT1, OUT_ROOT2, OUT_ROOT3]):
        html_files = list(src.rglob("*.htm"))
        batchsize = 2000
        for i in range(0, len(html_files), batchsize):
            batch = html_files[i:i+batchsize]
            with ProcessPoolExecutor(max_workers=4, initializer=driver_init) as pool:
                for saved_to in pool.map(process_file, batch, repeat(src, len(batch)), repeat(out, len(batch)), chunksize=1):
                    if saved_to:
                        print(saved_to)
            print("Restarting Selenium drivers...")
