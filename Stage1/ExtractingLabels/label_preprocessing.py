from swde_label_extraction import label_extraction, load_json_of_swde_file
from pathlib import Path
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import subprocess
import csv
import time

# ── paths ───────────────────────────────────────────────────────────────────────
SRC_FOLDER1 = Path("./data/swde/sourceCode/sourceCode/movie/movie")
SRC_FOLDER2 = Path("./data/swde/sourceCode/sourceCode/nbaplayer/nbaplayer")
SRC_FOLDER3 = Path("./data/swde/sourceCode/sourceCode/university/university")
OUT_ROOT1   = Path("./data/swde_HTMLgraphs/movie/movie")
OUT_ROOT2   = Path("./data/swde_HTMLgraphs/nbaplayer/nbaplayer")
OUT_ROOT3   = Path("./data/swde_HTMLgraphs/university/university")
OUT_ROOT1.mkdir(parents=True, exist_ok=True)
OUT_ROOT2.mkdir(parents=True, exist_ok=True)
OUT_ROOT3.mkdir(parents=True, exist_ok=True)
# SRC_FOLDER_WDC = Path("./data/wdc_microdata_html")
# OUT_ROOT_WDC = Path("./data/wdc_microdata_HTMLgraphs")
# OUT_ROOT_WDC.mkdir(parents=True, exist_ok=True)

# ── worker ──────────────────────────────────────────────────────────────────────
def process_file(filepath: Path, SRC: Path, OUT: Path, jsonAnswers=None) -> str | None:
    # build a parallel directory structure under OUT_ROOT
    rel = filepath.relative_to(SRC)
    out_dir = (OUT / rel).with_suffix("")

    # if (out_dir/"labels.npz").exists(): OVERWRITE
    #     return f"{out_dir} already written"

    out_dir.mkdir(parents=True, exist_ok=True)

    #This is slow, try to load this once per run, not per worker
    if not jsonAnswers:
        jsonAnswers = load_json_of_swde_file(str(filepath))

    try:   
        results, _, _, _ = label_extraction(filepath, jsonAnswers, out_dir, save=True, verifyTreeAgainstFile=True)

        for result in results:
            if isinstance(result, str):
                with open('./data/missedLabels.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([filepath, result])

        return f"Saved: {str(out_dir)}"

    except Exception as e:
        with open('./data/skipped.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([filepath, e])
        print(f"{filepath.absolute().resolve()}: {e}")
        return None

# ── main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for src, out in zip([SRC_FOLDER1, SRC_FOLDER2, SRC_FOLDER3],[OUT_ROOT1, OUT_ROOT2, OUT_ROOT3]):
        html_files = list(src.rglob("*.htm"))
        batchsize = len(html_files)
        workers = 8
        for i in range(0, len(html_files), batchsize):
            batch = html_files[i:i+batchsize]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for saved_to in pool.map(process_file, batch, repeat(src, len(batch)), repeat(out, len(batch)), chunksize=1):
                    if saved_to:
                        print(saved_to)
                    else:
                        print("Error, skipped and restarted Chrome...")