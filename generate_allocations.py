from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_SEED = 42
DEFAULT_OUTPUT_PATH = "data/allocations.csv"
DEFAULT_ADVISORS_PATH = Path("data/advisors.csv")
DEFAULT_MARKET_PATH = Path("data/market_returns.csv")
DEFAULT_ARCHETYPES_PATH = Path("archetypes.yaml")
DEFAULT_CATEGORIES_PATH = Path("categories.yaml")
DEFAULT_VEHICLES_PATH = Path("vehicles.yaml")

ASSET_CLASS_BASE_WEIGHTS: dict[str, dict[str, float]] = {
    "conservative_income": {
        "equity": 0.18,
        "fixed_income": 0.52,
        "allocation": 0.22,
        "alternatives": 0.08,
    },
    "balanced_planner": {
        "equity": 0.36,
        "fixed_income": 0.28,
        "allocation": 0.26,
        "alternatives": 0.10,
    },
    "growth_allocator": {
        "equity": 0.54,
        "fixed_income": 0.14,
        "allocation": 0.22,
        "alternatives": 0.10,
    },
    "active_trader": {
        "equity": 0.56,
        "fixed_income": 0.08,
        "allocation": 0.10,
        "alternatives": 0.26,
    },
    "alternative_specialist": {
        "equity": 0.24,
        "fixed_income": 0.24,
        "allocation": 0.12,
        "alternatives": 0.40,
    },
    "fixed_income_specialist": {
        "equity": 0.14,
        "fixed_income": 0.62,
        "allocation": 0.16,
        "alternatives": 0.08,
    },
}

CHANNEL_ALLOCATION_TILTS: dict[str, dict[str, float]] = {
    "ria": {"alternatives": 0.01, "fixed_income": -0.01},
    "wirehouse": {"equity": 0.02, "allocation": -0.01, "fixed_income": -0.01},
    "independent_broker_dealer": {"allocation": 0.01, "equity": -0.01},
    "regional_broker_dealer": {"fixed_income": 0.01, "equity": -0.01},
    "private_bank": {"alternatives": 0.03, "fixed_income": -0.02, "allocation": -0.01},
    "insurance_broker_dealer": {"allocation": 0.02, "equity": -0.01, "alternatives": -0.01},
}

TENURE_ALLOCATION_TILTS: dict[str, dict[str, float]] = {
    "emerging": {"equity": 0.01, "fixed_income": -0.01},
    "established": {},
    "tenured": {"fixed_income": 0.01, "equity": -0.01},
}

CONCENTRATION_ALPHA: dict[str, float] = {
    "conservative_income": 2.8,
    "balanced_planner": 2.2,
    "growth_allocator": 1.8,
    "active_trader": 1.3,
    "alternative_specialist": 1.6,
    "fixed_income_specialist": 2.4,
}

REBALANCE_STRENGTH: dict[str, float] = {
    "low": 0.22,
    "medium": 0.14,
    "high": 0.08,
}

FLOW_PERSISTENCE_DRAG: dict[str, float] = {
    "high": 0.03,
    "medium": 0.00,
    "low": -0.03,
}

VOLATILITY_REBALANCE_BONUS: dict[str, float] = {
    "low": -0.02,
    "medium": 0.00,
    "high": 0.05,
}

REGIME_SHIFT_MAP: dict[str, dict[str, float]] = {
    "risk_on": {
        "equity": 0.025,
        "fixed_income": -0.020,
        "alternatives": 0.005,
        "allocation": -0.010,
    },
    "rate_shock": {
        "equity": -0.010,
        "fixed_income": -0.020,
        "alternatives": 0.022,
        "allocation": 0.008,
    },
    "risk_off": {
        "equity": -0.040,
        "fixed_income": 0.030,
        "allocation": 0.012,
        "alternatives": -0.002,
    },
    "choppy_sideways": {
        "equity": -0.006,
        "fixed_income": 0.004,
        "allocation": 0.004,
        "alternatives": -0.002,
    },
}

REGIME_STYLE_BONUS: dict[str, dict[str, float]] = {
    "risk_on": {
        "growth": 0.020,
        "opportunistic": 0.018,
        "floating_rate": 0.006,
        "defensive": -0.012,
        "duration": -0.006,
    },
    "rate_shock": {
        "growth": -0.022,
        "inflation_sensitive": 0.022,
        "floating_rate": 0.016,
        "duration": -0.024,
        "defensive": 0.006,
    },
    "risk_off": {
        "defensive": 0.016,
        "duration": 0.018,
        "absolute_return": 0.010,
        "growth": -0.018,
        "opportunistic": -0.020,
    },
    "choppy_sideways": {
        "flexible": 0.010,
        "absolute_return": 0.010,
        "balanced": 0.004,
        "growth": -0.006,
    },
}

VEHICLE_CHANNEL_TILTS: dict[str, dict[str, float]] = {
    "ria": {"etf": 1.35, "sma": 1.20, "variable_annuity": 0.65},
    "wirehouse": {"mutual_fund": 1.15, "sma": 1.20, "etf": 1.10},
    "independent_broker_dealer": {"mutual_fund": 1.20, "variable_annuity": 1.20},
    "regional_broker_dealer": {"mutual_fund": 1.15, "etf": 0.90},
    "private_bank": {"sma": 1.55, "interval_fund": 1.25, "variable_annuity": 0.60},
    "insurance_broker_dealer": {"variable_annuity": 1.75, "mutual_fund": 1.15, "etf": 0.75},
}

VEHICLE_ARCHETYPE_TILTS: dict[str, dict[str, float]] = {
    "conservative_income": {"variable_annuity": 1.30, "mutual_fund": 1.10, "etf": 0.90},
    "balanced_planner": {"mutual_fund": 1.10, "etf": 1.05},
    "growth_allocator": {"etf": 1.25, "sma": 1.10},
    "active_trader": {"etf": 1.45, "mutual_fund": 0.80, "variable_annuity": 0.60},
    "alternative_specialist": {"interval_fund": 1.65, "sma": 1.10},
    "fixed_income_specialist": {"mutual_fund": 1.15, "cit": 1.20},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic advisor allocations by month and category."
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
        "--vehicles-path",
        type=Path,
        default=DEFAULT_VEHICLES_PATH,
        help=f"Vehicles configuration path (default: {DEFAULT_VEHICLES_PATH}).",
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


def normalize(weights: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    clipped = np.clip(weights, floor, None)
    total = clipped.sum()
    if total <= 0:
        raise ValueError("Unable to normalize non-positive weights.")
    return clipped / total


def apply_tilts(base_mix: dict[str, float], tilt_map: dict[str, float]) -> dict[str, float]:
    adjusted = {key: base_mix.get(key, 0.0) + tilt_map.get(key, 0.0) for key in base_mix}
    weights = normalize(np.array(list(adjusted.values()), dtype=float))
    return dict(zip(adjusted.keys(), weights))


def build_category_lookup(categories: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    category_ids = [category["id"] for category in categories]
    lookup = {category["id"]: category for category in categories}
    return category_ids, lookup


def category_score(
    category: dict[str, Any],
    archetype: dict[str, Any],
    asset_class_target: float,
) -> float:
    score = asset_class_target
    preferred_assets = set(archetype.get("preferred_asset_classes", []))
    underweight_assets = set(archetype.get("underweight_asset_classes", []))
    preferred_categories = set(archetype.get("preferred_categories", []))

    if category["asset_class"] in preferred_assets:
        score *= 1.55
    if category["asset_class"] in underweight_assets:
        score *= 0.55
    if category["id"] in preferred_categories:
        score *= 1.70

    risk_level = int(category["risk_level"])
    risk_tolerance = archetype["risk_tolerance"]
    if risk_tolerance in {"low", "low_medium"}:
        score *= max(0.45, 1.10 - 0.12 * (risk_level - 2))
    elif risk_tolerance == "medium":
        score *= max(0.55, 1.05 - 0.05 * abs(risk_level - 3))
    elif risk_tolerance == "medium_high":
        score *= 0.95 + 0.06 * (risk_level - 3)
    elif risk_tolerance == "high":
        score *= 0.90 + 0.10 * (risk_level - 3)

    if category["style"] == "defensive":
        score *= 1.10 if risk_tolerance in {"low", "low_medium"} else 0.95
    if category["style"] == "absolute_return":
        score *= 1.15 if archetype["id"] == "alternative_specialist" else 1.0

    return max(score, 1e-6)


def strategic_weights_for_advisor(
    advisor: pd.Series,
    archetype: dict[str, Any],
    categories: list[dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    base_mix = ASSET_CLASS_BASE_WEIGHTS[advisor["archetype_id"]]
    base_mix = apply_tilts(base_mix, CHANNEL_ALLOCATION_TILTS.get(advisor["channel"], {}))
    base_mix = apply_tilts(base_mix, TENURE_ALLOCATION_TILTS.get(advisor["tenure_band"], {}))

    category_scores = np.array(
        [
            category_score(
                category=category,
                archetype=archetype,
                asset_class_target=base_mix[category["asset_class"]],
            )
            for category in categories
        ],
        dtype=float,
    )

    alpha = CONCENTRATION_ALPHA[advisor["archetype_id"]]
    dirichlet_draw = rng.dirichlet(category_scores * alpha)
    return normalize(dirichlet_draw)


def regime_target_weights(
    strategic_weights: np.ndarray,
    categories: list[dict[str, Any]],
    regime_id: str,
) -> np.ndarray:
    shift_map = REGIME_SHIFT_MAP.get(regime_id, {})
    style_bonus = REGIME_STYLE_BONUS.get(regime_id, {})
    tilted = []
    for base_weight, category in zip(strategic_weights, categories):
        multiplier = 1.0
        multiplier += shift_map.get(category["asset_class"], 0.0)
        multiplier += style_bonus.get(category["style"], 0.0)
        tilted.append(base_weight * max(multiplier, 0.2))
    return normalize(np.array(tilted, dtype=float))


def rebalanced_weights(
    prior_weights: np.ndarray,
    strategic_weights: np.ndarray,
    monthly_returns: np.ndarray,
    categories: list[dict[str, Any]],
    turnover: str,
    flow_persistence: str,
    regime_id: str,
    volatility_regime: str,
    rng: np.random.Generator,
) -> np.ndarray:
    drifted = normalize(prior_weights * (1.0 + monthly_returns))
    target = regime_target_weights(strategic_weights, categories, regime_id)

    rebalance_speed = REBALANCE_STRENGTH[turnover]
    rebalance_speed += FLOW_PERSISTENCE_DRAG[flow_persistence]
    rebalance_speed += VOLATILITY_REBALANCE_BONUS[volatility_regime]
    rebalance_speed = float(np.clip(rebalance_speed, 0.04, 0.35))

    noise = rng.normal(0.0, 0.0025, size=len(prior_weights))
    updated = drifted * (1.0 - rebalance_speed) + target * rebalance_speed + noise
    return normalize(updated)


def eligible_vehicles(
    category: dict[str, Any], vehicles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        vehicle
        for vehicle in vehicles
        if category["asset_class"] in set(vehicle["eligible_asset_classes"])
    ]


def vehicle_scores(
    advisor: pd.Series,
    archetype: dict[str, Any],
    category: dict[str, Any],
    vehicles: list[dict[str, Any]],
) -> np.ndarray:
    scores = []
    channel_tilts = VEHICLE_CHANNEL_TILTS.get(advisor["channel"], {})
    archetype_tilts = VEHICLE_ARCHETYPE_TILTS.get(advisor["archetype_id"], {})

    for vehicle in vehicles:
        vehicle_id = vehicle["id"]
        score = float(vehicle["default_weight"])
        score *= channel_tilts.get(vehicle_id, 1.0)
        score *= archetype_tilts.get(vehicle_id, 1.0)

        if category["style"] == "opportunistic" and vehicle_id == "etf":
            score *= 1.25
        if category["style"] == "absolute_return" and vehicle_id == "interval_fund":
            score *= 1.20
        if category["style"] == "outcome_oriented" and vehicle_id in {"cit", "variable_annuity"}:
            score *= 1.20
        if category["asset_class"] == "alternatives" and vehicle_id == "mutual_fund":
            score *= 0.85

        scores.append(max(score, 1e-6))

    return np.array(scores, dtype=float)


def vehicle_split_for_category(
    advisor: pd.Series,
    archetype: dict[str, Any],
    category: dict[str, Any],
    vehicles: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[tuple[str, float]]:
    allowed_vehicles = eligible_vehicles(category, vehicles)
    scores = vehicle_scores(advisor, archetype, category, allowed_vehicles)
    shares = rng.dirichlet(scores * 18.0)

    if len(shares) > 2:
        shares = np.where(shares < 0.02, 0.0, shares)
        shares = normalize(shares)

    return [
        (vehicle["id"], float(share))
        for vehicle, share in zip(allowed_vehicles, shares)
        if share > 0
    ]


def build_allocations(
    advisors: pd.DataFrame,
    market: pd.DataFrame,
    archetypes: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    archetype_lookup = {item["id"]: item for item in archetypes}
    category_ids, _ = build_category_lookup(categories)

    months = sorted(market["month"].unique().tolist())
    monthly_market = {
        month: market.loc[market["month"] == month]
        .set_index("mstar_category")
        .reindex(category_ids)
        for month in months
    }

    rows: list[dict[str, Any]] = []
    advisor_groups = advisors.sort_values("advisor_id").itertuples(index=False)

    for advisor in advisor_groups:
        advisor_series = pd.Series(advisor._asdict())
        archetype = archetype_lookup[advisor_series["archetype_id"]]
        strategic = strategic_weights_for_advisor(advisor_series, archetype, categories, rng)
        vehicle_splits = {
            category["id"]: vehicle_split_for_category(
                advisor_series, archetype, category, vehicles, rng
            )
            for category in categories
        }
        current_weights = strategic.copy()
        starting_aum = float(advisor_series["aum"])

        for month_idx, month in enumerate(months):
            month_market = monthly_market[month]
            month_returns = month_market["return"].to_numpy(dtype=float)

            if month_idx > 0:
                current_weights = rebalanced_weights(
                    prior_weights=current_weights,
                    strategic_weights=strategic,
                    monthly_returns=month_returns,
                    categories=categories,
                    turnover=advisor_series["turnover"],
                    flow_persistence=advisor_series["flow_persistence"],
                    regime_id=str(month_market["market_regime"].iloc[0]),
                    volatility_regime=str(month_market["volatility_regime"].iloc[0]),
                    rng=rng,
                )

            category_assets = current_weights * starting_aum
            for idx, category in enumerate(categories):
                market_row = month_market.iloc[idx]
                category_weight = float(current_weights[idx])
                category_asset_value = float(category_assets[idx])
                for vehicle_id, vehicle_share in vehicle_splits[category["id"]]:
                    rows.append(
                        {
                            "advisor_id": advisor_series["advisor_id"],
                            "month": month,
                            "mstar_category": category["id"],
                            "vehicle": vehicle_id,
                            "allocation_pct": round(category_weight * vehicle_share, 8),
                            "category_allocation_pct": round(category_weight, 8),
                            "vehicle_allocation_pct": round(vehicle_share, 8),
                            "category_assets": round(category_asset_value * vehicle_share, 2),
                            "market_regime": market_row["market_regime"],
                            "volatility_regime": market_row["volatility_regime"],
                            "category_return": round(float(month_returns[idx]), 6),
                        }
                    )

    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    advisors = pd.read_csv(args.advisors_path)
    market = pd.read_csv(args.market_path)
    archetypes = load_yaml_list(args.archetypes_path, "archetypes")
    categories = load_yaml_list(args.categories_path, "categories")
    vehicles = load_yaml_list(args.vehicles_path, "vehicles")

    allocations = build_allocations(
        advisors=advisors,
        market=market,
        archetypes=archetypes,
        categories=categories,
        vehicles=vehicles,
        seed=args.seed,
    )
    write_csv(allocations, args.output)
    print(
        f"Wrote {len(allocations)} allocation rows to {args.output} using seed={args.seed}. "
        f"Advisors={allocations['advisor_id'].nunique()}, months={allocations['month'].nunique()}, "
        f"categories={allocations['mstar_category'].nunique()}, vehicles={allocations['vehicle'].nunique()}."
    )


if __name__ == "__main__":
    main()
