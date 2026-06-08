from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_SEED = 42
DEFAULT_OUTPUT_PATH = "data/monthly_flows.csv"
DEFAULT_ADVISORS_PATH = Path("data/advisors.csv")
DEFAULT_ALLOCATIONS_PATH = Path("data/allocations.csv")
DEFAULT_MARKET_PATH = Path("data/market_returns.csv")
DEFAULT_ARCHETYPES_PATH = Path("archetypes.yaml")
DEFAULT_CATEGORIES_PATH = Path("categories.yaml")

CHANNEL_FLOW_MULTIPLIERS: dict[str, float] = {
    "ria": 1.05,
    "wirehouse": 1.15,
    "independent_broker_dealer": 1.00,
    "regional_broker_dealer": 0.95,
    "private_bank": 0.92,
    "insurance_broker_dealer": 0.90,
}

TENURE_GROSS_MULTIPLIERS: dict[str, float] = {
    "emerging": 1.18,
    "established": 1.00,
    "tenured": 0.90,
}

TENURE_REDEMPTION_MULTIPLIERS: dict[str, float] = {
    "emerging": 0.88,
    "established": 1.00,
    "tenured": 1.08,
}

REGIME_GROSS_MULTIPLIERS: dict[str, float] = {
    "risk_on": 1.18,
    "rate_shock": 0.95,
    "risk_off": 0.72,
    "choppy_sideways": 0.92,
}

REGIME_REDEMPTION_MULTIPLIERS: dict[str, float] = {
    "risk_on": 0.90,
    "rate_shock": 1.08,
    "risk_off": 1.28,
    "choppy_sideways": 1.02,
}

VOLATILITY_GROSS_MULTIPLIERS: dict[str, float] = {
    "low": 1.05,
    "medium": 1.00,
    "high": 0.85,
}

VOLATILITY_REDEMPTION_MULTIPLIERS: dict[str, float] = {
    "low": 0.96,
    "medium": 1.00,
    "high": 1.12,
}

ASSET_CLASS_GROSS_TILTS: dict[str, float] = {
    "equity": 1.06,
    "fixed_income": 0.97,
    "real_assets": 0.98,
    "allocation": 1.01,
    "alternatives": 0.82,
}

ASSET_CLASS_REDEMPTION_TILTS: dict[str, float] = {
    "equity": 1.02,
    "fixed_income": 0.96,
    "real_assets": 1.03,
    "allocation": 0.98,
    "alternatives": 1.08,
}

STYLE_GROSS_TILTS: dict[str, float] = {
    "growth": 1.05,
    "value": 1.01,
    "core": 1.00,
    "income": 0.97,
    "inflation_sensitive": 1.02,
    "defensive": 0.95,
    "duration": 0.96,
    "spread": 1.01,
    "floating_rate": 1.02,
    "flexible": 1.00,
    "balanced": 1.00,
    "outcome_oriented": 0.98,
    "opportunistic": 0.90,
    "absolute_return": 0.88,
}

STYLE_REDEMPTION_TILTS: dict[str, float] = {
    "growth": 1.04,
    "value": 0.99,
    "core": 1.00,
    "income": 0.98,
    "inflation_sensitive": 1.03,
    "defensive": 0.95,
    "duration": 0.97,
    "spread": 1.02,
    "floating_rate": 0.99,
    "flexible": 1.00,
    "balanced": 0.99,
    "outcome_oriented": 0.98,
    "opportunistic": 1.08,
    "absolute_return": 0.92,
}

TURNOVER_GROSS_BOOST: dict[str, float] = {
    "low": 0.90,
    "medium": 1.00,
    "high": 1.12,
}

TURNOVER_REDEMPTION_BOOST: dict[str, float] = {
    "low": 0.94,
    "medium": 1.00,
    "high": 1.10,
}

FLOW_PERSISTENCE_FLOW_STABILITY: dict[str, float] = {
    "high": 0.80,
    "medium": 1.00,
    "low": 1.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate monthly advisor flows and assets by category."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--advisors-path",
        type=Path,
        default=DEFAULT_ADVISORS_PATH,
        help=f"Advisor master input path (default: {DEFAULT_ADVISORS_PATH}).",
    )
    parser.add_argument(
        "--allocations-path",
        type=Path,
        default=DEFAULT_ALLOCATIONS_PATH,
        help=f"Allocation input path (default: {DEFAULT_ALLOCATIONS_PATH}).",
    )
    parser.add_argument(
        "--market-path",
        type=Path,
        default=DEFAULT_MARKET_PATH,
        help=f"Market returns input path (default: {DEFAULT_MARKET_PATH}).",
    )
    parser.add_argument(
        "--archetypes-path",
        type=Path,
        default=DEFAULT_ARCHETYPES_PATH,
        help=f"Archetypes configuration path (default: {DEFAULT_ARCHETYPES_PATH}).",
    )
    parser.add_argument(
        "--categories-path",
        type=Path,
        default=DEFAULT_CATEGORIES_PATH,
        help=f"Categories configuration path (default: {DEFAULT_CATEGORIES_PATH}).",
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


def preferred_multiplier(
    category_id: str,
    asset_class: str,
    archetype: dict[str, Any],
) -> float:
    multiplier = 1.0
    if asset_class in set(archetype.get("preferred_asset_classes", [])):
        multiplier *= 1.18
    if asset_class in set(archetype.get("underweight_asset_classes", [])):
        multiplier *= 0.84
    if category_id in set(archetype.get("preferred_categories", [])):
        multiplier *= 1.22
    return multiplier


def calculate_gross_rate(
    allocation_pct: float,
    category_return: float,
    market_regime: str,
    volatility_regime: str,
    advisor: dict[str, Any],
    category: dict[str, Any],
    preference_multiplier: float,
    rng: np.random.Generator,
) -> float:
    rate = 0.0105
    rate *= float(advisor["gross_sales_bias"])
    rate *= CHANNEL_FLOW_MULTIPLIERS[str(advisor["channel"])]
    rate *= TENURE_GROSS_MULTIPLIERS[str(advisor["tenure_band"])]
    rate *= REGIME_GROSS_MULTIPLIERS[market_regime]
    rate *= VOLATILITY_GROSS_MULTIPLIERS[volatility_regime]
    rate *= ASSET_CLASS_GROSS_TILTS[str(category["asset_class"])]
    rate *= STYLE_GROSS_TILTS[str(category["style"])]
    rate *= TURNOVER_GROSS_BOOST[str(advisor["turnover"])]
    rate *= preference_multiplier

    if category_return > 0:
        rate *= 1.0 + min(category_return * 3.5, 0.18)
    else:
        rate *= 1.0 + max(category_return * 2.0, -0.12)

    allocation_strength = 0.70 + min(allocation_pct * 1.5, 0.55)
    rate *= allocation_strength

    noise_scale = 0.18 * FLOW_PERSISTENCE_FLOW_STABILITY[str(advisor["flow_persistence"])]
    rate *= max(0.55, 1.0 + rng.normal(0.0, noise_scale))
    return float(np.clip(rate, 0.0006, 0.055))


def calculate_redemption_rate(
    allocation_pct: float,
    category_return: float,
    market_regime: str,
    volatility_regime: str,
    advisor: dict[str, Any],
    category: dict[str, Any],
    preference_multiplier: float,
    rng: np.random.Generator,
) -> float:
    rate = 0.0095
    rate *= float(advisor["redemption_sensitivity"])
    rate *= TENURE_REDEMPTION_MULTIPLIERS[str(advisor["tenure_band"])]
    rate *= REGIME_REDEMPTION_MULTIPLIERS[market_regime]
    rate *= VOLATILITY_REDEMPTION_MULTIPLIERS[volatility_regime]
    rate *= ASSET_CLASS_REDEMPTION_TILTS[str(category["asset_class"])]
    rate *= STYLE_REDEMPTION_TILTS[str(category["style"])]
    rate *= TURNOVER_REDEMPTION_BOOST[str(advisor["turnover"])]

    rate *= 1.0 / max(preference_multiplier, 0.70)

    if category_return < 0:
        rate *= 1.0 + min(abs(category_return) * 4.5, 0.30)
    else:
        rate *= 1.0 - min(category_return * 2.0, 0.10)

    concentration_pressure = 0.92 + min(allocation_pct * 0.85, 0.20)
    rate *= concentration_pressure

    noise_scale = 0.16 * FLOW_PERSISTENCE_FLOW_STABILITY[str(advisor["flow_persistence"])]
    rate *= max(0.60, 1.0 + rng.normal(0.0, noise_scale))
    return float(np.clip(rate, 0.0005, 0.060))


def build_monthly_flows(
    advisors: pd.DataFrame,
    allocations: pd.DataFrame,
    archetypes: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    advisor_lookup = advisors.set_index("advisor_id").to_dict(orient="index")
    archetype_lookup = {item["id"]: item for item in archetypes}
    category_lookup = {item["id"]: item for item in categories}
    preference_lookup: dict[tuple[str, str], float] = {}
    for archetype_id, archetype in archetype_lookup.items():
        for category_id, category in category_lookup.items():
            preference_lookup[(archetype_id, category_id)] = preferred_multiplier(
                category_id, str(category["asset_class"]), archetype
            )

    working = allocations.copy()
    working["month"] = pd.to_datetime(working["month"])
    working = working.sort_values(["advisor_id", "month", "mstar_category"]).reset_index(
        drop=True
    )

    rows: list[dict[str, Any]] = []
    prior_asset_map: dict[tuple[str, str], float] = {}

    for record in working.itertuples(index=False):
        advisor = advisor_lookup[record.advisor_id]
        category = category_lookup[str(record.mstar_category)]
        key = (record.advisor_id, record.mstar_category)
        allocation_pct = float(record.allocation_pct)
        category_return = float(record.category_return)
        market_regime = str(record.market_regime)
        volatility_regime = str(record.volatility_regime)
        preference = preference_lookup[(str(advisor["archetype_id"]), str(record.mstar_category))]

        gross_rate = calculate_gross_rate(
            allocation_pct,
            category_return,
            market_regime,
            volatility_regime,
            advisor,
            category,
            preference,
            rng,
        )
        redemption_rate = calculate_redemption_rate(
            allocation_pct,
            category_return,
            market_regime,
            volatility_regime,
            advisor,
            category,
            preference,
            rng,
        )

        prior_asset_value = prior_asset_map.get(key, float(record.category_assets))
        gross_sales = prior_asset_value * gross_rate
        redemptions = prior_asset_value * redemption_rate
        net_sales = gross_sales - redemptions
        assets = prior_asset_value * (1.0 + category_return) + net_sales
        assets = max(assets, prior_asset_value * 0.10, 0.0)
        prior_asset_map[key] = assets

        rows.append(
            {
                "advisor_id": record.advisor_id,
                "month": record.month.strftime("%Y-%m-01"),
                "mstar_category": record.mstar_category,
                "gross_sales": round(gross_sales, 2),
                "redemptions": round(redemptions, 2),
                "net_sales": round(net_sales, 2),
                "assets": round(assets, 2),
                "allocation_pct": round(allocation_pct, 8),
                "market_regime": market_regime,
                "volatility_regime": volatility_regime,
                "category_return": round(category_return, 6),
            }
        )

    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    advisors = pd.read_csv(args.advisors_path)
    allocations = pd.read_csv(args.allocations_path)
    archetypes = load_yaml_list(args.archetypes_path, "archetypes")
    categories = load_yaml_list(args.categories_path, "categories")

    monthly_flows = build_monthly_flows(
        advisors=advisors,
        allocations=allocations,
        archetypes=archetypes,
        categories=categories,
        seed=args.seed,
    )
    write_csv(monthly_flows, args.output)
    print(
        f"Wrote {len(monthly_flows)} flow rows to {args.output} using seed={args.seed}. "
        f"Advisors={monthly_flows['advisor_id'].nunique()}, months={monthly_flows['month'].nunique()}, "
        f"categories={monthly_flows['mstar_category'].nunique()}."
    )


if __name__ == "__main__":
    main()
