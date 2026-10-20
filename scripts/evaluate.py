import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rareseek.__main__ import main
if __name__ == "__main__":
    sys.argv.insert(1, "evaluate")
    main()
