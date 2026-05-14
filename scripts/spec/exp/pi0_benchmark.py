import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.spec.pi0_benchmark import main


if __name__ == "__main__":
    main()
