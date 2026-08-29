from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "Data_processed"
CONFIG_DIR = ROOT / "config"

INPUT_FILE = PROCESSED_DIR / "driver_features.parquet"
OUTPUT_FILE = PROCESSED_DIR / "materiality_events.parquet"

BUSINESS_CONFIG = CONFIG_DIR / "business_config.yaml"


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum relative movement to consider.
# This is NOT sufficient by itself to declare materiality.
MIN_RELATIVE_CHANGE = 0.08

Z_SCORE_MEDIUM = 2.0
Z_SCORE_HIGH = 2.5
Z_SCORE_CRITICAL = 3.0

BUSINESS_IMPACT_MEDIUM = 50_000
BUSINESS_IMPACT_HIGH = 75_000
BUSINESS_IMPACT_CRITICAL = 100_000
# KPIs we currently monitor.
MONITORED_KPIS = [
    "revenue",
    "orders",
    "aov",
    "conversion_rate",
    "marketing_spend",
    "inventory_availability",
    "delivery_delay_rate",
    "return_rate",
    "discount_rate",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_features():

    print("\nLoading driver features...")

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["region", "date"]
    ).reset_index(drop=True)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    return df


# ============================================================
# LOAD BUSINESS CONFIG
# ============================================================

def load_business_config():

    if not BUSINESS_CONFIG.exists():

        print(
            "\nWARNING: business_config.yaml not found."
        )

        return {}

    with open(
        BUSINESS_CONFIG,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    print("\nBusiness configuration loaded.")

    return config or {}


# ============================================================
# HISTORICAL VOLATILITY
# ============================================================

def calculate_z_scores(df):

    print("\nCalculating statistical deviations...")

    # We use the historical rolling standard deviation.
    #
    # IMPORTANT:
    # We shift the rolling window by one day.
    #
    # This prevents today's movement from influencing the
    # baseline against which today's movement is evaluated.

    for kpi in MONITORED_KPIS:

        change_col = f"{kpi}_pct_change_1d"

        if change_col not in df.columns:
            continue

        grouped = df.groupby("region")[change_col]

        historical_mean = (
            grouped
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    30,
                    min_periods=14
                )
                .mean()
            )
        )

        historical_std = (
            grouped
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    30,
                    min_periods=14
                )
                .std()
            )
        )

        df[f"{kpi}_historical_change_mean"] = (
            historical_mean
        )

        df[f"{kpi}_historical_change_std"] = (
            historical_std
        )

        df[f"{kpi}_z_score"] = (
            (
                df[change_col]
                - historical_mean
            )
            / historical_std.replace(0, np.nan)
        )

    return df


# ============================================================
# BUSINESS IMPACT
# ============================================================

def calculate_business_impact(df):

    print("\nCalculating business impact...")

    # Revenue has a direct monetary interpretation.
    #
    # For revenue:
    #
    #     impact = actual revenue - expected revenue
    #
    # A negative value means revenue is below baseline.

    if "revenue_rolling_mean_7d" in df.columns:

        df["revenue_business_impact"] = (
            df["revenue"]
            - df["revenue_rolling_mean_7d"]
        )

        df["revenue_absolute_impact"] = (
            df["revenue_business_impact"]
            .abs()
        )

    else:

        df["revenue_business_impact"] = np.nan

        df["revenue_absolute_impact"] = np.nan

    return df


# ============================================================
# MATERIALITY FLAGS
# ============================================================

def create_materiality_flags(df):

    print("\nCreating materiality flags...")

    revenue_change = (
        df["revenue_pct_change_1d"]
        .abs()
    )

    revenue_z = (
        df["revenue_z_score"]
        .abs()
    )

    business_impact = (
        df["revenue_absolute_impact"]
    )

    # --------------------------------------------------------
    # Relative movement
    # --------------------------------------------------------

    df["material_relative_change"] = (
        revenue_change >= MIN_RELATIVE_CHANGE
    )

    # --------------------------------------------------------
    # Statistical movement
    # --------------------------------------------------------

    df["material_statistical_change"] = (
        revenue_z >= Z_SCORE_MEDIUM
    )

    # --------------------------------------------------------
    # Business impact
    # --------------------------------------------------------

    df["material_business_impact"] = (
        business_impact >= BUSINESS_IMPACT_MEDIUM
    )

    # --------------------------------------------------------
    # Overall materiality
    # --------------------------------------------------------
    #
    # We require:
    #
    #     meaningful movement
    #          AND
    #     statistical OR business significance
    #
    # This prevents a huge-looking percentage movement on
    # an insignificant business amount from automatically
    # becoming a major event.

    df["is_material"] = (
        df["material_relative_change"]
        &
        (
            df["material_statistical_change"]
            |
            df["material_business_impact"]
        )
    )

    return df


# ============================================================
# MATERIALITY SCORE
# ============================================================

def calculate_materiality_score(df):

    print("\nCalculating materiality score...")

    # Normalize the statistical component.
    statistical_score = (
        df["revenue_z_score"]
        .abs()
        .clip(0, 4)
        / 4
    )

    # Normalize business impact.
    business_score = (
        df["revenue_absolute_impact"]
        .fillna(0)
        .clip(
            0,
            BUSINESS_IMPACT_CRITICAL
        )
        / BUSINESS_IMPACT_CRITICAL
    )

    # Relative movement.
    relative_score = (
        df["revenue_pct_change_1d"]
        .abs()
        .clip(0, 0.50)
        / 0.50
    )

    # Weighted deterministic score.
    #
    # Statistical unusualness gets the highest weight.
    # Business impact gets the second highest.
    # Raw percentage movement gets the lowest.

    df["materiality_score"] = (
        0.45 * statistical_score
        +
        0.35 * business_score
        +
        0.20 * relative_score
    )

    return df


# ============================================================
# PRIORITY
# ============================================================

def assign_priority(df):

    print("\nAssigning event priority...")

    def priority(row):

        if not row["is_material"]:
            return "NONE"

        z = abs(row["revenue_z_score"])
        impact = row["revenue_absolute_impact"]

        if (
            z >= Z_SCORE_CRITICAL
            and impact >= BUSINESS_IMPACT_CRITICAL
        ):
            return "CRITICAL"

        if (
            z >= Z_SCORE_HIGH
            or impact >= BUSINESS_IMPACT_HIGH
        ):
            return "HIGH"

        if (
            z >= Z_SCORE_MEDIUM
            or impact >= BUSINESS_IMPACT_MEDIUM
        ):
            return "MEDIUM"

        return "LOW"

    df["priority"] = df.apply(
        priority,
        axis=1
    )

    return df


# ============================================================
# REASON CODES
# ============================================================

def assign_reason_codes(df):

    print("\nGenerating materiality reason codes...")

    def reasons(row):

        reasons = []

        if row["material_relative_change"]:
            reasons.append(
                "large_relative_movement"
            )

        if row["material_statistical_change"]:
            reasons.append(
                "statistically_unusual"
            )

        if row["material_business_impact"]:
            reasons.append(
                "material_business_impact"
            )

        return reasons

    df["materiality_reasons"] = df.apply(
        reasons,
        axis=1
    )

    return df


# ============================================================
# EVENT EXTRACTION
# ============================================================

def extract_events(df):

    events = df[
        df["is_material"]
    ].copy()

    # Keep the event table relatively compact.
    columns = [
        "date",
        "region",

        "revenue",
        "revenue_pct_change_1d",
        "revenue_rolling_mean_7d",
        "revenue_vs_7d_baseline_pct",

        "revenue_historical_change_mean",
        "revenue_historical_change_std",
        "revenue_z_score",

        "revenue_business_impact",
        "revenue_absolute_impact",

        "material_relative_change",
        "material_statistical_change",
        "material_business_impact",

        "materiality_score",
        "priority",
        "materiality_reasons",
    ]

    available = [
        c for c in columns
        if c in events.columns
    ]

    events = events[available]

    events = events.sort_values(
        [
            "date",
            "materiality_score"
        ],
        ascending=[
            False,
            False
        ]
    )

    return events


# ============================================================
# SAVE
# ============================================================

def save_events(events):

    events.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✓ Saved materiality events:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  Material events: {len(events):,}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(events):

    print("\n" + "=" * 70)
    print("MATERIALITY SUMMARY")
    print("=" * 70)

    if events.empty:

        print("\nNo material KPI movements detected.")

        return

    print("\nPriority distribution:")

    print(
        events["priority"]
        .value_counts()
        .to_string()
    )

    print("\nTop material events:")

    display_columns = [
        "date",
        "region",
        "revenue",
        "revenue_pct_change_1d",
        "revenue_z_score",
        "revenue_absolute_impact",
        "materiality_score",
        "priority",
    ]

    available = [
        c
        for c in display_columns
        if c in events.columns
    ]

    print(
        events[available]
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("MATERIALITY DETECTOR")
    print("=" * 70)

    df = load_features()

    load_business_config()

    df = calculate_z_scores(df)

    df = calculate_business_impact(df)

    df = create_materiality_flags(df)

    df = calculate_materiality_score(df)

    df = assign_priority(df)

    df = assign_reason_codes(df)

    events = extract_events(df)

    save_events(events)

    print_summary(events)

    print("\n" + "=" * 70)
    print("MATERIALITY DETECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()