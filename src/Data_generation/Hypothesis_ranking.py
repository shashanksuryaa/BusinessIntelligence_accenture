# src/Data_generation/hypothesis_ranking.py

import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PROCESSED = os.path.join(BASE_DIR, "Data_processed")

INPUT_FILE = os.path.join(
    DATA_PROCESSED,
    "validated_hypotheses.parquet"
)

OUTPUT_FILE = os.path.join(
    DATA_PROCESSED,
    "ranked_hypotheses.parquet"
)


# Weights for final evidence score
WEIGHTS = {
    "historical_support": 0.35,
    "analytic_evidence": 0.25,
    "recurrence": 0.15,
    "lag_consistency": 0.10,
    "direction_consistency": 0.10,
    "segment_consistency": 0.05,
}


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(series):
    """Convert a column to numeric safely."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def normalize_strength(value):
    """
    Convert evidence strength into a numeric score.
    """
    mapping = {
        "HIGH": 1.0,
        "STRONG": 1.0,
        "MODERATE": 0.65,
        "MEDIUM": 0.65,
        "WEAK": 0.35,
        "LOW": 0.20,
        "NONE": 0.0,
    }

    return mapping.get(str(value).upper(), 0.0)


def calculate_driver_priority(row):
    """
    Additional business-oriented priority.

    Direct contributors such as orders and AOV are important,
    but underlying operational drivers can also receive high
    priority when their evidence is strong.
    """

    driver = str(row.get("driver", "")).lower()

    direct_contributors = {
        "orders": 1.00,
        "aov": 0.95,
    }

    underlying_drivers = {
        "conversion_rate": 1.00,
        "marketing_spend": 0.90,
        "inventory_availability": 0.85,
        "delivery_delay_rate": 0.85,
        "return_rate": 0.75,
        "discount_rate": 0.65,
    }

    if driver in direct_contributors:
        return direct_contributors[driver]

    if driver in underlying_drivers:
        return underlying_drivers[driver]

    return 0.50


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading validated hypotheses...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows loaded    : {len(df)}")
    print(f"Columns loaded : {len(df.columns)}")

    return df


# ============================================================
# PREPARE EVIDENCE
# ============================================================

def prepare_evidence(df):

    print("\nPreparing evidence scores...")

    # Historical support
    if "historical_support_score" in df.columns:
        historical = safe_numeric(
            df["historical_support_score"]
        ).clip(0, 1)
    else:
        historical = pd.Series(
            0.0,
            index=df.index
        )

    # Analytic evidence
    if "evidence_score" in df.columns:
        analytic = safe_numeric(
            df["evidence_score"]
        ).clip(0, 1)
    else:
        analytic = pd.Series(
            0.0,
            index=df.index
        )

    # Historical recurrence
    recurrence = (
        safe_numeric(
            df.get(
                "recurrence_rate",
                pd.Series(0.0, index=df.index)
            )
        )
        .clip(0, 1)
    )

    # Lag consistency
    lag_consistency = (
        safe_numeric(
            df.get(
                "lag_consistency",
                pd.Series(0.0, index=df.index)
            )
        )
        .clip(0, 1)
    )

    # Direction consistency
    direction_consistency = (
        safe_numeric(
            df.get(
                "direction_consistency",
                pd.Series(0.0, index=df.index)
            )
        )
        .clip(0, 1)
    )

    # Segment consistency
    segment_consistency = (
        safe_numeric(
            df.get(
                "segment_consistency",
                pd.Series(0.0, index=df.index)
            )
        )
        .clip(0, 1)
    )

    # Final evidence score
    final_score = (
        historical * WEIGHTS["historical_support"]
        + analytic * WEIGHTS["analytic_evidence"]
        + recurrence * WEIGHTS["recurrence"]
        + lag_consistency * WEIGHTS["lag_consistency"]
        + direction_consistency * WEIGHTS["direction_consistency"]
        + segment_consistency * WEIGHTS["segment_consistency"]
    )

    df["ranking_evidence_score"] = final_score

    df["business_driver_priority"] = df.apply(
        calculate_driver_priority,
        axis=1
    )

    # Combine statistical evidence with business importance
    df["final_hypothesis_score"] = (
        0.85 * df["ranking_evidence_score"]
        + 0.15 * df["business_driver_priority"]
    )

    return df


# ============================================================
# ASSIGN RANKS
# ============================================================

def assign_ranks(df):

    print("Assigning hypothesis ranks...")

    # Rank hypotheses within each event.
    #
    # An event is identified by date + region.
    #
    # This prevents a driver from South from competing directly
    # with a driver from West.

    group_columns = []

    if "date" in df.columns:
        group_columns.append("date")

    if "region" in df.columns:
        group_columns.append("region")

    if group_columns:

        df["hypothesis_rank"] = (
            df.groupby(group_columns)["final_hypothesis_score"]
            .rank(
                ascending=False,
                method="first"
            )
        )

    else:

        df["hypothesis_rank"] = (
            df["final_hypothesis_score"]
            .rank(
                ascending=False,
                method="first"
            )
        )

    df["hypothesis_rank"] = (
        df["hypothesis_rank"]
        .astype(int)
    )

    return df


# ============================================================
# CLASSIFY HYPOTHESES
# ============================================================

def classify_hypotheses(df):

    print("Classifying hypothesis strength...")

    def classify(score):

        if score >= 0.80:
            return "HIGH"

        elif score >= 0.60:
            return "MODERATE"

        elif score >= 0.40:
            return "WEAK"

        else:
            return "LOW"

    df["ranking_strength"] = (
        df["final_hypothesis_score"]
        .apply(classify)
    )

    # Whether this driver should actually be surfaced
    # to the next layer.

    def investigation_status(row):

        validation = str(
            row.get(
                "hypothesis_status",
                ""
            )
        )

        score = row["final_hypothesis_score"]

        if validation == "SUPPORTED_FOR_INVESTIGATION":

            if score >= 0.60:
                return "PRIORITIZE"

            elif score >= 0.40:
                return "SECONDARY"

        return "DO_NOT_PRIORITIZE"

    df["investigation_status"] = df.apply(
        investigation_status,
        axis=1
    )

    return df


# ============================================================
# SELECT TOP HYPOTHESES
# ============================================================

def mark_top_hypotheses(df):

    print("Identifying top hypotheses...")

    group_columns = []

    if "date" in df.columns:
        group_columns.append("date")

    if "region" in df.columns:
        group_columns.append("region")

    if group_columns:

        df["is_top_hypothesis"] = (
            df["hypothesis_rank"] <= 3
        )

    else:

        df["is_top_hypothesis"] = (
            df["hypothesis_rank"] <= 3
        )

    return df


# ============================================================
# SAVE
# ============================================================

def save_results(df):

    # Sort for easy inspection
    sort_columns = []

    if "date" in df.columns:
        sort_columns.append("date")

    if "region" in df.columns:
        sort_columns.append("region")

    sort_columns.append("hypothesis_rank")

    df = df.sort_values(
        sort_columns
    ).reset_index(drop=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✓ Saved ranked hypotheses:\n"
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  Rows : {len(df)}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(df):

    print("\n" + "=" * 70)
    print("HYPOTHESIS RANKING SUMMARY")
    print("=" * 70)

    if "ranking_strength" in df.columns:

        print("\nRanking strength:")

        print(
            df["ranking_strength"]
            .value_counts()
        )

    if "investigation_status" in df.columns:

        print("\nInvestigation status:")

        print(
            df["investigation_status"]
            .value_counts()
        )

    # Top hypotheses
    columns = [
        "date",
        "region",
        "driver",
        "hypothesis_rank",
        "final_hypothesis_score",
        "ranking_strength",
        "investigation_status",
    ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    if columns:

        print("\nTop ranked hypotheses:")

        print(
            df[
                df["hypothesis_rank"] <= 3
            ][columns]
            .head(30)
            .to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("HYPOTHESIS RANKING COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HYPOTHESIS RANKER")
    print("=" * 70)

    df = load_data()

    df = prepare_evidence(df)

    df = assign_ranks(df)

    df = classify_hypotheses(df)

    df = mark_top_hypotheses(df)

    save_results(df)

    print_summary(df)


if __name__ == "__main__":
    main()