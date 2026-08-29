# src/Analytics/analytic_agent.py

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_DIR = os.path.join(BASE_DIR, "Data_processed")

MATERIALITY_FILE = os.path.join(
    DATA_DIR, "materiality_events.parquet"
)

DRIVER_FEATURE_FILE = os.path.join(
    DATA_DIR, "driver_features.parquet"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR, "driver_analysis.parquet"
)

TARGET = "revenue"

DRIVERS = [
    "orders",
    "aov",
    "conversion_rate",
    "marketing_spend",
    "inventory_availability",
    "delivery_delay_rate",
    "return_rate",
    "discount_rate",
]

# These are the historical windows we want to test.
LAGS = [1, 3, 7]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_corr(x, y):
    """
    Pearson correlation with protection against:
      - missing values
      - constant series
      - insufficient observations

    IMPORTANT:
    This is descriptive evidence only.
    It is NOT causal evidence.
    """

    valid = pd.concat([x, y], axis=1).dropna()

    if len(valid) < 10:
        return np.nan

    if valid.iloc[:, 0].nunique() <= 1:
        return np.nan

    if valid.iloc[:, 1].nunique() <= 1:
        return np.nan

    return valid.iloc[:, 0].corr(valid.iloc[:, 1])


def correlation_strength(corr):
    """
    Convert correlation magnitude into an interpretable bucket.

    This is descriptive evidence, NOT causal evidence.
    """

    if pd.isna(corr):
        return "INSUFFICIENT"

    value = abs(corr)

    if value >= 0.70:
        return "STRONG"

    if value >= 0.40:
        return "MODERATE"

    if value >= 0.20:
        return "WEAK"

    return "VERY_WEAK"


def normalized_strength(corr):
    """
    Convert absolute correlation into a 0-1 score.
    """

    if pd.isna(corr):
        return 0.0

    return min(abs(corr), 1.0)


def direction_alignment(
    revenue_change,
    driver_change,
    historical_correlation,
):
    """
    Determine whether the observed driver movement is
    directionally consistent with its historical relationship
    with revenue.

    Example:
        correlation(revenue, delivery_delay) = -0.50
        delivery_delay increases
        revenue decreases

    => CONSISTENT_WITH_HISTORY

    This is descriptive evidence only.
    It does NOT establish causality.
    """

    if (
        pd.isna(revenue_change)
        or pd.isna(driver_change)
        or pd.isna(historical_correlation)
    ):
        return "UNKNOWN"

    if (
        revenue_change == 0
        or driver_change == 0
        or historical_correlation == 0
    ):
        return "NEUTRAL"

    # Historical relationship tells us how a driver movement
    # would be expected to affect revenue.
    expected_revenue_direction = (
        np.sign(driver_change)
        * np.sign(historical_correlation)
    )

    actual_revenue_direction = np.sign(
        revenue_change
    )

    if (
        expected_revenue_direction
        == actual_revenue_direction
    ):
        return "CONSISTENT_WITH_HISTORY"

    return "INCONSISTENT_WITH_HISTORY"


# ============================================================
# GLOBAL DRIVER ANALYSIS
# ============================================================

def calculate_global_correlations(df):
    """
    Calculate same-period and lagged correlations between
    revenue and candidate drivers.

    IMPORTANT:
    Lagging is performed WITHIN EACH REGION so that a lag
    does not accidentally connect the last row of one region
    to the first row of another region.

    This is descriptive analysis only.
    """

    results = []

    working = df.sort_values(
        ["region", "date"]
    ).copy()

    for driver in DRIVERS:

        if driver not in working.columns:
            continue

        same_corr = safe_corr(
            working[TARGET],
            working[driver],
        )

        row = {
            "driver": driver,
            "correlation_same_day": same_corr,
        }

        for lag in LAGS:

            shifted_driver = (
                working.groupby("region")[driver]
                .shift(lag)
            )

            lag_corr = safe_corr(
                working[TARGET],
                shifted_driver,
            )

            row[f"correlation_lag_{lag}d"] = lag_corr

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# SEGMENT CONSISTENCY
# ============================================================

def calculate_segment_consistency(df):
    """
    Check whether the revenue-driver relationship is
    reasonably consistent across regions.

    segment_mean_abs_correlation:
        Average magnitude of regional correlations.

    segment_positive_share:
        Fraction of regions where the relationship is positive.

    This is descriptive evidence only.
    """

    results = []

    if "region" not in df.columns:
        return pd.DataFrame()

    for driver in DRIVERS:

        if driver not in df.columns:
            continue

        regional_corrs = []

        for region, group in df.groupby("region"):

            corr = safe_corr(
                group[TARGET],
                group[driver],
            )

            if not pd.isna(corr):
                regional_corrs.append(corr)

        if len(regional_corrs) == 0:

            consistency = np.nan
            positive_share = np.nan

        else:

            consistency = np.mean(
                np.abs(regional_corrs)
            )

            positive_share = np.mean(
                np.array(regional_corrs) > 0
            )

        results.append({
            "driver": driver,
            "segment_count": len(regional_corrs),
            "segment_mean_abs_correlation": consistency,
            "segment_positive_share": positive_share,
        })

    return pd.DataFrame(results)


# ============================================================
# BEST LAG
# ============================================================

def select_best_lag(
    same_corr,
    lag_1,
    lag_3,
    lag_7,
):
    """
    Select the lag with the strongest absolute correlation.

    Lag 0 (same day) is included.

    Returns:
        best_lag
        best_lag_corr

    IMPORTANT:
    We choose by absolute correlation magnitude because both
    positive and negative relationships can be meaningful.
    """

    lag_values = {
        0: same_corr,
        1: lag_1,
        3: lag_3,
        7: lag_7,
    }

    valid_lags = {
        lag: corr
        for lag, corr in lag_values.items()
        if pd.notna(corr)
    }

    if not valid_lags:
        return np.nan, np.nan

    best_lag = max(
        valid_lags,
        key=lambda lag: abs(valid_lags[lag]),
    )

    best_lag_corr = valid_lags[best_lag]

    return best_lag, best_lag_corr


# ============================================================
# EVENT-LEVEL ANALYSIS
# ============================================================

def analyze_event(
    event,
    df,
    global_corrs,
    segment_consistency,
):
    """
    Analyze candidate drivers for one material event.

    The output is a ranked analytical evidence layer.
    It is NOT a causal inference layer.
    """

    event_date = pd.to_datetime(
        event["date"]
    )

    region = event["region"]

    # --------------------------------------------------------
    # Locate event row
    # --------------------------------------------------------

    event_rows = df[
        (pd.to_datetime(df["date"]) == event_date)
        & (df["region"] == region)
    ]

    if len(event_rows) == 0:
        return []

    event_row = event_rows.iloc[0]

    # --------------------------------------------------------
    # Previous day
    # --------------------------------------------------------

    previous_rows = df[
        (pd.to_datetime(df["date"]) < event_date)
        & (df["region"] == region)
    ].sort_values("date")

    if len(previous_rows) == 0:
        previous_row = None
    else:
        previous_row = previous_rows.iloc[-1]

    # --------------------------------------------------------
    # Revenue movement
    # --------------------------------------------------------

    revenue_change = event_row.get(
        "revenue_pct_change_1d",
        np.nan,
    )

    results = []

    for driver in DRIVERS:

        if driver not in df.columns:
            continue

        driver_value = event_row.get(
            driver,
            np.nan,
        )

        # ----------------------------------------------------
        # Driver local movement
        # ----------------------------------------------------

        if previous_row is not None:

            previous_value = previous_row.get(
                driver,
                np.nan,
            )

            if (
                pd.notna(driver_value)
                and pd.notna(previous_value)
                and previous_value != 0
            ):

                driver_change = (
                    driver_value - previous_value
                ) / abs(previous_value)

            else:
                driver_change = np.nan

        else:
            driver_change = np.nan

        # ----------------------------------------------------
        # Global correlation
        # ----------------------------------------------------

        corr_row = global_corrs[
            global_corrs["driver"] == driver
        ]

        if len(corr_row):

            corr_row = corr_row.iloc[0]

            same_corr = corr_row[
                "correlation_same_day"
            ]

            lag_1 = corr_row[
                "correlation_lag_1d"
            ]

            lag_3 = corr_row[
                "correlation_lag_3d"
            ]

            lag_7 = corr_row[
                "correlation_lag_7d"
            ]

        else:

            same_corr = np.nan
            lag_1 = np.nan
            lag_3 = np.nan
            lag_7 = np.nan

        # ----------------------------------------------------
        # Best lag
        # ----------------------------------------------------

        best_lag, best_lag_corr = select_best_lag(
            same_corr,
            lag_1,
            lag_3,
            lag_7,
        )

        # ----------------------------------------------------
        # Segment consistency
        # ----------------------------------------------------

        seg_row = segment_consistency[
            segment_consistency["driver"] == driver
        ]

        if len(seg_row):

            seg_row = seg_row.iloc[0]

            segment_consistency_score = seg_row[
                "segment_mean_abs_correlation"
            ]

            segment_positive_share = seg_row[
                "segment_positive_share"
            ]

        else:

            segment_consistency_score = np.nan
            segment_positive_share = np.nan

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        alignment = direction_alignment(
            revenue_change,
            driver_change,
            same_corr,
        )

        # ----------------------------------------------------
        # Evidence score
        # ----------------------------------------------------
        #
        # This is NOT probability.
        #
        # It combines:
        #   1. same-day correlation strength
        #   2. strongest lag correlation
        #   3. regional consistency
        #   4. current movement consistency
        #
        # Historical validation will later provide stronger
        # evidence before any causal-ish conclusion is made.
        # ----------------------------------------------------

        corr_score = normalized_strength(
            same_corr
        )

        lag_score = normalized_strength(
            best_lag_corr
        )

        segment_score = (
            segment_consistency_score
            if pd.notna(segment_consistency_score)
            else 0.0
        )

        # IMPORTANT:
        # Inconsistent movement must NOT receive the same
        # evidence credit as consistent movement.
        if alignment == "CONSISTENT_WITH_HISTORY":

            alignment_score = 1.0

        elif alignment == "INCONSISTENT_WITH_HISTORY":

            alignment_score = 0.0

        else:

            alignment_score = 0.0

        evidence_score = (
            0.35 * corr_score
            + 0.30 * lag_score
            + 0.20 * segment_score
            + 0.15 * alignment_score
        )

        # ----------------------------------------------------
        # Evidence bucket
        # ----------------------------------------------------

        if evidence_score >= 0.70:

            evidence_strength = "STRONG"

        elif evidence_score >= 0.45:

            evidence_strength = "MODERATE"

        elif evidence_score >= 0.25:

            evidence_strength = "WEAK"

        else:

            evidence_strength = "LOW"

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "date": event_date,

            "region": region,

            "target_kpi": TARGET,

            "driver": driver,

            "revenue_change": revenue_change,

            "driver_change": driver_change,

            "driver_value": driver_value,

            "direction_alignment": alignment,

            "correlation_same_day": same_corr,

            "correlation_lag_1d": lag_1,

            "correlation_lag_3d": lag_3,

            "correlation_lag_7d": lag_7,

            "best_lag_days": best_lag,

            "best_lag_correlation": best_lag_corr,

            "segment_consistency": (
                segment_consistency_score
            ),

            "segment_positive_share": (
                segment_positive_share
            ),

            "correlation_strength": (
                correlation_strength(same_corr)
            ),

            "evidence_score": evidence_score,

            "evidence_strength": evidence_strength,
        })

    return results


# ============================================================
# RANK DRIVERS
# ============================================================

def rank_drivers(results):
    """
    Rank drivers within each material event.
    """

    results = results.copy()

    results["driver_rank"] = (
        results
        .groupby(
            ["date", "region"]
        )["evidence_score"]
        .rank(
            method="dense",
            ascending=False,
        )
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ANALYTIC AGENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load materiality events
    # --------------------------------------------------------

    print("\nLoading materiality events...")

    materiality = pd.read_parquet(
        MATERIALITY_FILE
    )

    materiality["date"] = pd.to_datetime(
        materiality["date"]
    )

    # Only investigate actual material events.
    if "is_material" in materiality.columns:

        materiality = materiality[
            materiality["is_material"] == True
        ].copy()

    print(
        f"Material events loaded: {len(materiality)}"
    )

    # --------------------------------------------------------
    # Load driver features
    # --------------------------------------------------------

    print("\nLoading driver features...")

    df = pd.read_parquet(
        DRIVER_FEATURE_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        ["region", "date"]
    ).copy()

    print(
        f"Driver rows loaded: {len(df)}"
    )

    # --------------------------------------------------------
    # Global correlations
    # --------------------------------------------------------

    print(
        "\nCalculating global correlations..."
    )

    global_corrs = calculate_global_correlations(
        df
    )

    # --------------------------------------------------------
    # Segment consistency
    # --------------------------------------------------------

    print(
        "Calculating regional segment consistency..."
    )

    segment_consistency = (
        calculate_segment_consistency(df)
    )

    # --------------------------------------------------------
    # Event analysis
    # --------------------------------------------------------

    print(
        "\nAnalyzing material events..."
    )

    all_results = []

    for _, event in materiality.iterrows():

        event_results = analyze_event(
            event,
            df,
            global_corrs,
            segment_consistency,
        )

        all_results.extend(
            event_results
        )

    if not all_results:

        print(
            "\nNo driver analysis results generated."
        )

        return

    results = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    results = rank_drivers(
        results
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results = results.sort_values(
        [
            "date",
            "region",
            "driver_rank",
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ANALYTIC SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDriver analysis rows : {len(results)}"
    )

    print(
        f"Material events      : "
        f"{results[['date', 'region']].drop_duplicates().shape[0]}"
    )

    print(
        "\nEvidence distribution:"
    )

    print(
        results[
            "evidence_strength"
        ].value_counts()
    )

    print(
        "\nTop candidate drivers:"
    )

    top_drivers = (
        results[
            results["driver_rank"] == 1
        ][
            [
                "date",
                "region",
                "driver",
                "evidence_score",
                "evidence_strength",
                "correlation_same_day",
                "best_lag_days",
                "best_lag_correlation",
                "direction_alignment",
            ]
        ]
        .head(20)
    )

    print(
        top_drivers.to_string(
            index=False
        )
    )

    print(
        "\nGlobal correlations:"
    )

    print(
        global_corrs.to_string(
            index=False
        )
    )

    print(
        "\n✓ Saved driver analysis:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANALYTIC AGENT COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
