"""
Shim CLI — unika konfliktu nazwy z wbudowanym modułem `statistics` w notebookach.

Uruchomienie:
  python statistics.py
  python statistics.py --plot --episodes 100
"""

from rl_statistics import main

if __name__ == "__main__":
    main()
