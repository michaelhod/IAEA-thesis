from HTMLtoGraph import html_to_graph
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from seleniumDriver import driver_init, get_Driver
from scipy import sparse

# ── paths ───────────────────────────────────────────────────────────────────────
SRC_FOLDER = Path("./data/swde/sourceCode/sourceCode/movie")
OUT_ROOT   = Path("./data/swde_HTMLgraphs/movie")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── worker ──────────────────────────────────────────────────────────────────────
def process_file(filepath: Path) -> str | None:
    try:
        html = filepath.read_text(encoding="utf-8")
        A, X, E = html_to_graph(html, get_Driver())

        # build a parallel directory structure under OUT_ROOT
        rel   = filepath.relative_to(SRC_FOLDER)
        out_dir = (OUT_ROOT / rel).with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # save arrays
        np.save(out_dir / "A.npy", A)
        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "E.npy", E)

        return str(out_dir)

    except Exception as e:
        print(f"{filepath}: {e}")
        return None

# ── main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    html_files = list(SRC_FOLDER.rglob("*.htm"))

    with ProcessPoolExecutor(max_workers=4, initializer=driver_init) as pool:
        for saved_to in pool.map(process_file, html_files, chunksize=1):
            if saved_to:
                print(f"Saved: {saved_to}")
