"""Drive the legacy PaperBroker demo and print a ledger summary.

Thin orchestration only; all logic lives under ``quant``.
"""

from __future__ import annotations

from quant.app import format_paper_demo_plain, run_paper_demo


def main() -> None:
    print(format_paper_demo_plain(run_paper_demo()))


if __name__ == "__main__":
    main()
