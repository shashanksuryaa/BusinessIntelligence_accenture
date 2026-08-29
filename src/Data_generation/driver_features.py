from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "Data_processed"

INPUT_FILE = PROCESSED_DIR / "regional_kpis.parquet"
OUTPUT_FILE = PROCESSED_DIR / "driver_features.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

# Drivers that can potentially explain revenue movement.
DRIVER_KPIS = [
    "orders",
    "aov",
    "conversion_rate",
    "marketing_spend",
    "inventory_availability",
    "delivery_delay_rate",
    "return_rate",
    "discount_rate",
]

# Historical windows used for trend/context features.
ROLLING_WINDOWS = [3, 7, 14]

# Lags that the analytic engine can investigate.
LAG_PERIODS = [1, 3, 7]


# ============================================================
# LOAD DATA
# ============================================================

def load_regional_kpis():

    print("\nLoading regional KPI data...")

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["region", "date"]
    ).reset_index(drop=True)

    print(
        f"Loaded {len(df):,} regional KPI rows."
    )

    return df


# ============================================================
# PERCENTAGE CHANGE FEATURES
# ============================================================

def add_change_features(df):

    print("\nCreating percentage-change features...")

    grouped = df.groupby("region")

    # Revenue change is especially important because
    # revenue is our primary target KPI.

    df["revenue_pct_change_1d"] = (
        grouped["revenue"]
        .pct_change(1)
    )

    for kpi in DRIVER_KPIS:

        if kpi not in df.columns:
            continue

        df[f"{kpi}_pct_change_1d"] = (
            grouped[kpi]
            .pct_change(1)
        )

    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def add_rolling_features(df):

    print("\nCreating rolling trend features...")

    grouped = df.groupby("region")

    for kpi in ["revenue"] + DRIVER_KPIS:

        if kpi not in df.columns:
            continue

        for window in ROLLING_WINDOWS:

            # Rolling mean
            df[
                f"{kpi}_rolling_mean_{window}d"
            ] = (
                grouped[kpi]
                .transform(
                    lambda x:
                    x.rolling(
                        window,
                        min_periods=window
                    ).mean()
                )
            )

            # Rolling standard deviation
            df[
                f"{kpi}_rolling_std_{window}d"
            ] = (
                grouped[kpi]
                .transform(
                    lambda x:
                    x.rolling(
                        window,
                        min_periods=window
                    ).std()
                )
            )

    return df


# ============================================================
# LAG FEATURES
# ============================================================

def add_lag_features(df):

    print("\nCreating lag features...")

    grouped = df.groupby("region")

    # We want the analytic engine to be able to ask:
    #
    # "Did the driver move BEFORE revenue moved?"
    #
    # rather than only:
    #
    # "Did the driver and revenue move together?"

    for kpi in DRIVER_KPIS:

        if kpi not in df.columns:
            continue

        for lag in LAG_PERIODS:

            df[
                f"{kpi}_lag_{lag}d"
            ] = (
                grouped[kpi]
                .shift(lag)
            )

            df[
                f"{kpi}_change_lag_{lag}d"
            ] = (
                grouped[kpi]
                .pct_change(lag)
            )

    return df


# ============================================================
# REVENUE BASELINE FEATURES
# ============================================================

def add_revenue_baseline_features(df):

    print("\nCreating revenue baseline features...")

    grouped = df.groupby("region")

    # 7-day historical baseline
    df["revenue_rolling_mean_7d"] = (
        grouped["revenue"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                7,
                min_periods=7
            )
            .mean()
        )
    )

    # Difference from historical baseline
    df["revenue_vs_7d_baseline_pct"] = (
        (
            df["revenue"]
            - df["revenue_rolling_mean_7d"]
        )
        / df["revenue_rolling_mean_7d"]
    )

    # 14-day baseline
    df["revenue_rolling_mean_14d"] = (
        grouped["revenue"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                14,
                min_periods=14
            )
            .mean()
        )
    )

    df["revenue_vs_14d_baseline_pct"] = (
        (
            df["revenue"]
            - df["revenue_rolling_mean_14d"]
        )
        / df["revenue_rolling_mean_14d"]
    )

    return df


# ============================================================
# REGION REVENUE CONTRIBUTION
# ============================================================

def add_region_revenue_contribution(df):

    print("\nCreating regional revenue contribution...")

    daily_total = (
        df.groupby("date")["revenue"]
        .transform("sum")
    )

    df["region_revenue_share"] = (
        df["revenue"]
        / daily_total
    )

    return df


# ============================================================
# DRIVER DIRECTION FEATURES
# ============================================================

def add_direction_features(df):

    print("\nCreating driver direction features...")

    # These are descriptive features.
    #
    # They do NOT mean that a positive movement caused
    # a positive revenue movement.

    for kpi in DRIVER_KPIS:

        change_column = (
            f"{kpi}_pct_change_1d"
        )

        if change_column not in df.columns:
            continue

        df[
            f"{kpi}_direction"
        ] = np.select(
            [
                df[change_column] > 0.01,
                df[change_column] < -0.01,
            ],
            [
                "up",
                "down",
            ],
            default="stable",
        )

    return df


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

def clean_features(df):

    print("\nCleaning feature values...")

    # Replace infinite values generated by percentage changes.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Keep the rows.
    #
    # Early rows naturally have missing rolling/lag features
    # because historical observations don't exist yet.
    #
    # We intentionally DO NOT fill them with zero because
    # zero would incorrectly imply that the driver had a
    # measured value of zero.

    return df


# ============================================================
# SAVE
# ============================================================

def save_features(df):

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✓ Saved driver features:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  Rows    : {len(df):,}"
    )

    print(
        f"  Columns : {len(df.columns):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("DRIVER FEATURE ENGINEERING")
    print("=" * 70)

    df = load_regional_kpis()

    df = add_change_features(df)

    df = add_rolling_features(df)

    df = add_lag_features(df)

    df = add_revenue_baseline_features(df)

    df = add_region_revenue_contribution(df)

    df = add_direction_features(df)

    df = clean_features(df)

    save_features(df)

    print("\n" + "=" * 70)
    print("DRIVER FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print("\nSample:")

    sample_columns = [
        "date",
        "region",
        "revenue",
        "orders",
        "conversion_rate",
        "marketing_spend",
        "delivery_delay_rate",
        "revenue_pct_change_1d",
        "orders_pct_change_1d",
        "conversion_rate_pct_change_1d",
        "marketing_spend_pct_change_1d",
        "delivery_delay_rate_pct_change_1d",
        "marketing_spend_lag_1d",
        "marketing_spend_lag_3d",
        "marketing_spend_lag_7d",
        "revenue_vs_7d_baseline_pct",
        "region_revenue_share",
    ]

    available = [
        column
        for column in sample_columns
        if column in df.columns
    ]

    print(
        df[available]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()