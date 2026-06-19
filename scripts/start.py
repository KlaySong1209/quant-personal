"""Start the local quant-personal menu.

Thin orchestration only; all behavior lives in ``quant.app``.
"""

from __future__ import annotations

import sys

from quant.app import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

