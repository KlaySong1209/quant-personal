"""Generate seeded synthetic OHLCV CSVs into ``data/example/``.

Thin orchestration only; all logic lives under ``quant``.
"""

from __future__ import annotations

from quant.app import generate_example_data


def main() -> None:
    for path in generate_example_data():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
