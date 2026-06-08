from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_PERIODS = 24
DEFAULT_SEED = 42
DEFAULT_START_MONTH = "2024-01-01"
DEFAULT_OUTPUT_PATH = "data/market_returns.csv"
CATEGORIES_PATH = Path("categories.yaml")
REGIMES_PATH = Path("market_regimes.yaml")

REGIME_START_WEIGHTS: dict[str, float] = {
    "risk_on": 0.35,
    "rate_shock": 0.20,
    "risk_off": 0.20,
    "choppy_sideways": 0.25,
}

REGIME_TRANSITIONS: dict[str, dict[str, float]] = {
    "risk_on": {
        "risk_on": 0.48,
        "choppy_sideways": 0.25,
        "rate_shock": 0.17,
        "risk_off": 0.10,
    },
    "rate_shock": {
        "rate_shock": 0.38,
        "choppy_sideways": 0.25,
        "risk_on": 0.18,
        "risk_off": 0.19,
    },
    "risk_off": {
        "risk_off": 0.42,
        "choppy_sideways": 0.26,
        "risk_on": 0.12,
        "rate_shock": 0.20,
    },
    "choppy_sideways": {
        "choppy_sideways": 0.38,
        "risk_on": 0.26,
        "rate_shock": 0.20,
        "risk_off": 0.16,
    },
}

VOLATILITY_MULTIPLIERS: dict[str, float] = {
    "low": 0.75,
    "medium": 1.00,
    "high": 1.35,
}

VOLATILITY_RETURN_PENALTY: dict[str, float] = {
    "low": 0.0015,
    "medium": 0.0,
    "high": -0.0035,
}

REGIME_BASE_RETURNS: dict[str, dict[str, float]] = {
    "risk_on": {
        "equity": 0.014,
        "fixed_income": 0.003,
        "allocation": 0.008,
        "alternatives": 0.005,
    },
    "rate_shock": {
        "equity": 0.002,
        "fixed_income": -0.008,
        "allocation": -0.001,
        "alternatives": 0.001,
    },
    "risk_off": {
        "equity": -0.021,
        "fixed_income": 0.010,
        "allocation": -0.008,
        "alternatives": -0.003,
    },
    "choppy_sideways": {
        "equity": 0.002,
        "fixed_income": 0.001,
        "allocation": 0.0015,
        "alternatives": 0.001,
    },
}

ASSET_CLASS_VOLATILITY: dict[str, float] = {
    "equity": 0.042,
    "fixed_income": 0.016,
    "allocation": 0.022,
    "alternatives": 0.026,
}

STYLE_TILTS: dict[str, float] = {
    "growth": 0.0025,
    "value": 0.0015,
    "core": 0.0,
    "income": 0.0010,
    "inflation_sensitive": 0.0035,
    "defensive": -0.0010,
    "duration": 0.0015,
    "spread": 0.0010,
    "floating_rate": 0.0015,
    "flexible": 0.0005,
    "balanced": 0.0,
    "outcome_oriented": -0.0005,
    "opportunistic": 0.0010,
    "absolute_return": -0.0005,
}

REGION_TILTS: dict[str, float] = {
    "us": 0.0005,
    "developed_ex_us": 0.0,
    "emerging_markets": 0.0010,
    "global": 0.0,
}

REGIME_ROTATION_TILTS: dict[str, dict[str, float]] = {
    "risk_on": {
        "growth": 0.0045,
        "value": 0.0015,
        "inflation_sensitive": 0.0025,
        "floating_rate": 0.0010,
        "defensive": -0.0015,
        "duration": -0.0010,
    },
    "rate_shock": {
        "growth": -0.0055,
        "value": 0.0025,
        "inflation_sensitive": 0.0045,
        "floating_rate": 0.0040,
        "duration": -0.0060,
        "defensive": 0.0010,
    },
    "risk_off": {
        "growth": -0.0040,
        "value": -0.0020,
        "inflation_sensitive": -0.0035,
        "floating_rate": -0.0010,
        "duration": 0.0045,
        "defensive": 0.0030,
        "absolute_return": 0.0015,
    },
    "choppy_sideways": {
        "growth": -0.0010,
        "value": 0.0010,
        "inflation_sensitive": 0.0005,
        "floating_rate": 0.0005,
        "duration": 0.0005,
        "defensive": 0.0015,
        "absolute_return": 0.0010,
    },
}

SPECIAL_CATEGORY_TILTS: dict[str, dict[str, float]] = {
    "risk_on": {
        "us_small_cap": 0.0055,
        "diversified_emerging_markets": 0.0045,
        "high_yield_bond": 0.0025,
        "options_trading": 0.0060,
        "market_neutral": -0.0010,
    },
    "rate_shock": {
        "us_long_government_bond": -0.0085,
        "us_short_term_bond": 0.0015,
        "bank_loan": 0.0040,
        "natural_resources": 0.0045,
        "global_real_estate": -0.0030,
    },
    "risk_off": {
        "us_long_government_bond": 0.0075,
        "us_intermediate_core_bond": 0.0030,
        "us_short_term_bond": 0.0020,
        "high_yield_bond": -0.0050,
        "options_trading": -0.0065,
        "market_neutral": 0.0020,
    },
    "choppy_sideways": {
        "tactical_allocation": 0.0020,
        "market_neutral": 0.0015,
        "target_date": 0.0005,
        "options_trading": -0.0015,
    },
}

RISK_LEVEL_TILT = 0.0012


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic monthly market/category returns."
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=DEFAULT_PERIODS,
        help=f"Number of monthly periods to generate (default: {DEFAULT_PERIODS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--start-month",
        type=str,
        default=DEFAULT_START_MONTH,
        help=f"First month in YYYY-MM-DD format (default: {DEFAULT_START_MONTH}).",
    )
    parser.add_argument(
        "--categories-path",
        type=Path,
        default=CATEGORIES_PATH,
        help=f"Path to category configuration (default: {CATEGORIES_PATH}).",
    )
    parser.add_argument(
        "--regimes-path",
        type=Path,
        default=REGIMES_PATH,
        help=f"Path to market regime configuration (default: {REGIMES_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help=f"CSV output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def load_yaml_list(path: Path, key: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    items = payload.get(key, [])
    if not items:
        raise ValueError(f"No '{key}' entries found in {path}.")
    return items


def normalize_weights(mapping: dict[str, float], keys: list[str]) -> np.ndarray:
    weights = np.array([mapping.get(key, 0.0) for key in keys], dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"Invalid weights for keys: {keys}")
    return weights / total


def choose_next_regime(
    rng: np.random.Generator, current_regime: str, regime_ids: list[str]
) -> str:
    transition_probs = normalize_weights(
        REGIME_TRANSITIONS.get(current_regime, {}), regime_ids
    )
    return str(rng.choice(regime_ids, p=transition_probs))


def regime_path(
    periods: int, regime_ids: list[str], rng: np.random.Generator
) -> list[str]:
    first_probs = normalize_weights(REGIME_START_WEIGHTS, regime_ids)
    path = [str(rng.choice(regime_ids, p=first_probs))]
    while len(path) < periods:
        path.append(choose_next_regime(rng, path[-1], regime_ids))
    return enforce_regime_coverage(path, regime_ids, rng)


def enforce_regime_coverage(
    path: list[str], regime_ids: list[str], rng: np.random.Generator
) -> list[str]:
    missing_regimes = [regime_id for regime_id in regime_ids if regime_id not in path]
    if not missing_regimes:
        return path

    adjusted_path = list(path)
    candidate_positions = list(range(2, max(len(path) - 2, 3)))
    rng.shuffle(candidate_positions)

    for missing_regime, position in zip(missing_regimes, candidate_positions):
        adjusted_path[position] = missing_regime

    return adjusted_path


def macro_signal(regime_id: str, month_idx: int) -> float:
    cycle = np.sin(month_idx / 2.4) * 0.0015
    shock_bias = 0.0
    if regime_id == "risk_on":
        shock_bias = 0.0015
    elif regime_id == "rate_shock":
        shock_bias = -0.0010
    elif regime_id == "risk_off":
        shock_bias = -0.0020
    return cycle + shock_bias


def category_expected_return(
    category: dict[str, Any], regime_id: str, month_idx: int
) -> float:
    asset_class = category["asset_class"]
    style = category["style"]
    category_id = category["id"]
    region = category["region"]
    risk_level = int(category["risk_level"])

    expected_return = REGIME_BASE_RETURNS[regime_id][asset_class]
    expected_return += STYLE_TILTS.get(style, 0.0)
    expected_return += REGION_TILTS.get(region, 0.0)
    expected_return += REGIME_ROTATION_TILTS.get(regime_id, {}).get(style, 0.0)
    expected_return += SPECIAL_CATEGORY_TILTS.get(regime_id, {}).get(category_id, 0.0)
    expected_return += (risk_level - 3) * RISK_LEVEL_TILT * 0.35
    expected_return += macro_signal(regime_id, month_idx)

    if asset_class == "allocation":
        expected_return *= 0.85
    elif asset_class == "alternatives":
        expected_return *= 0.75

    return expected_return


def category_realized_return(
    category: dict[str, Any],
    regime_id: str,
    month_idx: int,
    regime_volatility: str,
    asset_class_shocks: dict[str, float],
    rng: np.random.Generator,
) -> float:
    asset_class = category["asset_class"]
    expected = category_expected_return(category, regime_id, month_idx)
    volatility = (
        ASSET_CLASS_VOLATILITY[asset_class]
        * VOLATILITY_MULTIPLIERS[regime_volatility]
        * (0.85 + 0.05 * int(category["risk_level"]))
    )
    idiosyncratic_noise = rng.normal(0.0, volatility * 0.35)
    realized = (
        expected
        + asset_class_shocks[asset_class]
        + idiosyncratic_noise
        + VOLATILITY_RETURN_PENALTY[regime_volatility]
    )
    return float(np.clip(realized, -0.22, 0.18))


def build_market_history(
    periods: int,
    seed: int,
    start_month: str,
    categories_path: Path,
    regimes_path: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = load_yaml_list(categories_path, "categories")
    regimes = load_yaml_list(regimes_path, "market_regimes")
    regime_lookup = {regime["id"]: regime for regime in regimes}
    regime_ids = [regime["id"] for regime in regimes]
    months = pd.date_range(start=start_month, periods=periods, freq="MS")
    realized_regimes = regime_path(periods, regime_ids, rng)

    rows: list[dict[str, Any]] = []
    for month_idx, month in enumerate(months):
        regime_id = realized_regimes[month_idx]
        regime = regime_lookup[regime_id]
        volatility_regime = regime["volatility_level"]

        asset_class_shocks = {
            asset_class: rng.normal(
                loc=0.0,
                scale=ASSET_CLASS_VOLATILITY[asset_class]
                * VOLATILITY_MULTIPLIERS[volatility_regime]
                * 0.45,
            )
            for asset_class in ASSET_CLASS_VOLATILITY
        }

        for category in categories:
            monthly_return = category_realized_return(
                category=category,
                regime_id=regime_id,
                month_idx=month_idx,
                regime_volatility=volatility_regime,
                asset_class_shocks=asset_class_shocks,
                rng=rng,
            )
            rows.append(
                {
                    "month": month.strftime("%Y-%m-01"),
                    "market_regime": regime_id,
                    "volatility_regime": volatility_regime,
                    "flow_sentiment": regime["flow_sentiment"],
                    "mstar_category": category["id"],
                    "category_name": category["name"],
                    "asset_class": category["asset_class"],
                    "style": category["style"],
                    "region": category["region"],
                    "risk_level": int(category["risk_level"]),
                    "return": round(monthly_return, 6),
                }
            )

    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    market = build_market_history(
        periods=args.periods,
        seed=args.seed,
        start_month=args.start_month,
        categories_path=args.categories_path,
        regimes_path=args.regimes_path,
    )
    write_csv(market, args.output)
    print(
        f"Wrote {len(market)} market rows to {args.output} using seed={args.seed}. "
        f"Months={market['month'].nunique()}, categories={market['mstar_category'].nunique()}."
    )


if __name__ == "__main__":
    main()
