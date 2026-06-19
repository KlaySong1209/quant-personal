"""Entry point for ``python -m quant``."""

from __future__ import annotations

import sys

from quant.app import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

