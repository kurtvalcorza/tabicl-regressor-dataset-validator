"""DIMER Pipeline Builder entrypoint for the TabICLv2 regressor dataset validator.

The validation implementation lives in ``validator.py``; this is the literal
``validate.py`` entrypoint DIMER invokes. Keeping the implementation in
``validator.py`` preserves the existing test/import surface unchanged.
"""
from __future__ import annotations

import sys

from validator import main

if __name__ == "__main__":
    sys.exit(main())
