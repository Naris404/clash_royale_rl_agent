"""Bootstrap the src-layout package for legacy root entry points."""
from pathlib import Path
import sys

src = str(Path(__file__).resolve().parent / "src")
if src not in sys.path:
    sys.path.insert(0, src)
