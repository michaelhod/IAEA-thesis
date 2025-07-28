import subprocess, sys

def run(script):
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise SystemExit(result.returncode)

if __name__ == "__main__":
    run("./Stage1/ExtractingGraphs/SWDE_preprocessing.py")
    run("./Stage1/ExtractingLabels/label_preprocessing.py")
