from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_ADVISOR_COUNT = 1000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_PATH = "data/advisors.csv"
ARCHETYPES_PATH = Path("archetypes.yaml")

ARCHETYPE_WEIGHTS: dict[str, float] = {
    "conservative_income": 0.20,
    "balanced_planner": 0.30,
    "growth_allocator": 0.22,
    "active_trader": 0.10,
    "alternative_specialist": 0.08,
    "fixed_income_specialist": 0.10,
}

CHANNEL_WEIGHTS: dict[str, float] = {
    "ria": 0.20,
    "wirehouse": 0.30,
    "independent_broker_dealer": 0.24,
    "regional_broker_dealer": 0.12,
    "private_bank": 0.08,
    "insurance_broker_dealer": 0.06,
}

CHANNEL_AUM_MULTIPLIERS: dict[str, float] = {
    "ria": 1.10,
    "wirehouse": 1.30,
    "independent_broker_dealer": 0.95,
    "regional_broker_dealer": 0.90,
    "private_bank": 1.60,
    "insurance_broker_dealer": 0.85,
}

ARCHETYPE_AUM_MULTIPLIERS: dict[str, float] = {
    "conservative_income": 1.05,
    "balanced_planner": 1.00,
    "growth_allocator": 1.10,
    "active_trader": 0.90,
    "alternative_specialist": 1.20,
    "fixed_income_specialist": 0.95,
}

CHANNEL_ARCHETYPE_TILTS: dict[str, dict[str, float]] = {
    "ria": {
        "balanced_planner": 1.20,
        "growth_allocator": 1.20,
        "alternative_specialist": 1.25,
    },
    "wirehouse": {
        "growth_allocator": 1.15,
        "active_trader": 1.20,
        "conservative_income": 0.90,
    },
    "independent_broker_dealer": {
        "conservative_income": 1.15,
        "balanced_planner": 1.10,
    },
    "regional_broker_dealer": {
        "balanced_planner": 1.10,
        "fixed_income_specialist": 1.15,
    },
    "private_bank": {
        "conservative_income": 1.10,
        "alternative_specialist": 1.20,
        "active_trader": 0.80,
    },
    "insurance_broker_dealer": {
        "conservative_income": 1.30,
        "fixed_income_specialist": 1.10,
        "active_trader": 0.75,
    },
}

TENURE_OPTIONS = np.array(["emerging", "established", "tenured"])
TENURE_WEIGHTS = np.array([0.22, 0.46, 0.32])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic advisor master dataset."
    )
    parser.add_argument(
        "--num-advisors",
        type=int,
        default=DEFAULT_ADVISOR_COUNT,
        help=f"Number of advisors to generate (default: {DEFAULT_ADVISOR_COUNT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--archetypes-path",
        type=Path,
        default=ARCHETYPES_PATH,
        help=f"Path to archetype configuration (default: {ARCHETYPES_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help=f"CSV output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def load_archetypes(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    archetypes = payload.get("archetypes", [])
    if not archetypes:
        raise ValueError(f"No archetypes found in {path}.")
    return archetypes


def normalize_weights(mapping: dict[str, float], keys: list[str]) -> np.ndarray:
    weights = np.array([mapping.get(key, 0.0) for key in keys], dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"Invalid weight configuration for keys: {keys}")
    return weights / total


def choose_channel(rng: np.random.Generator, archetype_id: str) -> str:
    channels = list(CHANNEL_WEIGHTS)
    weights = np.array([CHANNEL_WEIGHTS[channel] for channel in channels], dtype=float)
    tilts = CHANNEL_ARCHETYPE_TILTS
    for idx, channel in enumerate(channels):
        weights[idx] *= tilts.get(channel, {}).get(archetype_id, 1.0)
    weights /= weights.sum()
    return str(rng.choice(channels, p=weights))


def generate_aum(
    rng: np.random.Generator,
    archetype_id: str,
    channel: str,
    tenure_band: str,
) -> float:
    base_aum = rng.lognormal(mean=np.log(120_000_000), sigma=1.0)
    scaled_aum = (
        base_aum
        * ARCHETYPE_AUM_MULTIPLIERS.get(archetype_id, 1.0)
        * CHANNEL_AUM_MULTIPLIERS.get(channel, 1.0)
    )

    if tenure_band == "emerging":
        scaled_aum *= rng.uniform(0.35, 0.70)
    elif tenure_band == "established":
        scaled_aum *= rng.uniform(0.85, 1.15)
    else:
        scaled_aum *= rng.uniform(1.05, 1.45)

    return float(np.clip(scaled_aum, 5_000_000, 12_000_000_000))


def generate_households(
    rng: np.random.Generator,
    aum: float,
    channel: str,
    tenure_band: str,
) -> int:
    channel_client_size = {
        "private_bank": 2_100_000,
        "wirehouse": 1_500_000,
        "ria": 1_350_000,
        "independent_broker_dealer": 1_150_000,
        "regional_broker_dealer": 1_050_000,
        "insurance_broker_dealer": 900_000,
    }
    avg_household_assets = channel_client_size.get(channel, 1_200_000)
    raw_households = aum / avg_household_assets

    if tenure_band == "emerging":
        raw_households *= rng.uniform(0.80, 1.00)
    elif tenure_band == "tenured":
        raw_households *= rng.uniform(1.00, 1.20)

    noise = rng.normal(loc=0.0, scale=max(raw_households * 0.08, 3.0))
    households = int(round(raw_households + noise))
    return max(households, 15)


def build_advisors(num_advisors: int, seed: int, archetypes_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    archetypes = load_archetypes(archetypes_path)
    archetype_ids = [item["id"] for item in archetypes]
    archetype_lookup = {item["id"]: item for item in archetypes}
    archetype_probs = normalize_weights(ARCHETYPE_WEIGHTS, archetype_ids)

    rows: list[dict[str, Any]] = []
    for advisor_num in range(1, num_advisors + 1):
        advisor_id = f"ADV{advisor_num:04d}"
        archetype_id = str(rng.choice(archetype_ids, p=archetype_probs))
        channel = choose_channel(rng, archetype_id)
        tenure_band = str(rng.choice(TENURE_OPTIONS, p=TENURE_WEIGHTS))
        aum = generate_aum(rng, archetype_id, channel, tenure_band)
        households = generate_households(rng, aum, channel, tenure_band)

        archetype = archetype_lookup[archetype_id]
        rows.append(
            {
                "advisor_id": advisor_id,
                "advisor_name": f"Advisor {advisor_num:04d}",
                "channel": channel,
                "archetype_id": archetype_id,
                "archetype_name": archetype["name"],
                "risk_tolerance": archetype["risk_tolerance"],
                "turnover": archetype["turnover"],
                "flow_persistence": archetype["flow_persistence"],
                "tenure_band": tenure_band,
                "household_count": households,
                "aum": round(aum, 2),
                "gross_sales_bias": float(archetype["gross_sales_bias"]),
                "redemption_sensitivity": float(archetype["redemption_sensitivity"]),
            }
        )

    advisors = pd.DataFrame(rows).sort_values("aum", ascending=False).reset_index(
        drop=True
    )
    return advisors


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    advisors = build_advisors(
        num_advisors=args.num_advisors,
        seed=args.seed,
        archetypes_path=args.archetypes_path,
    )
    write_csv(advisors, args.output)
    print(
        f"Wrote {len(advisors)} advisors to {args.output} using seed={args.seed}. "
        f"Median AUM=${advisors['aum'].median():,.0f}, max AUM=${advisors['aum'].max():,.0f}."
    )


if __name__ == "__main__":
    main()
