"""Convert a Kalodata/master workbook into the canonical MVP CSV.

Synthetic/default values are clearly limited to fields absent from the source.
They make the demo runnable but must be replaced before production training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import load_and_clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Kalodata/master CSV or XLSX")
    parser.add_argument("--output", default="data/creator_campaign.csv")
    args = parser.parse_args()

    dataset = load_and_clean(args.input)
    dataset["creator_cost"] = np.clip(dataset["followers"] * 25, 1_000_000, 100_000_000)
    effective_price = (dataset["product_price"] * (1 - dataset["discount_rate"])).clip(lower=1)
    dataset["orders"] = np.floor(dataset["revenue"] / effective_price)
    dataset["gmv"] = dataset["revenue"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    print(f"Wrote {len(dataset):,} rows to {output}")


if __name__ == "__main__":
    main()
