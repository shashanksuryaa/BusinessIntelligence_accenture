# src/Data_generation/evidence_builder.py

import os
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PROCESSED = os.path.join(BASE_DIR, "Data_processed")

MATERIALITY_FILE = os.path.join(
    DATA_PROCESSED,
    "materiality_events.parquet"
)

DRIVER_FILE = os.path.join(
    DATA_PROCESSED,
    "driver_features.parquet"
)

VALIDATED_FILE = os.path.join(
    DATA_PROCESSED,
    "validated_hypotheses.parquet"
)

RANKED_FILE = os.path.join(
    DATA_PROCESSED,
    "ranked_hypotheses.parquet"
)

OUTPUT_FILE = os.path.join(
    DATA_PROCESSED,
    "evidence_bundle.parquet"
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_value(row, column, default=None):
    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def load_parquet(path, name):

    print(f"Loading {name}...")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    df = pd.read_parquet(path)

    print(
        f"  Rows    : {len(df)}\n"
        f"  Columns : {len(df.columns)}"
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    materiality = load_parquet(
        MATERIALITY_FILE,
        "materiality events"
    )

    driver_features = load_parquet(
        DRIVER_FILE,
        "driver features"
    )

    validated = load_parquet(
        VALIDATED_FILE,
        "validated hypotheses"
    )

    ranked = load_parquet(
        RANKED_FILE,
        "ranked hypotheses"
    )

    return (
        materiality,
        driver_features,
        validated,
        ranked
    )


# ============================================================
# NORMALIZE KEYS
# ============================================================

def normalize_keys(df):

    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"]
        ).dt.date

    if "region" in df.columns:
        df["region"] = df["region"].astype(str)

    return df


# ============================================================
# FIND EVENT
# ============================================================

def get_event_row(
    materiality,
    date,
    region
):

    matches = materiality[
        (materiality["date"] == date)
        &
        (materiality["region"] == region)
    ]

    if len(matches) == 0:
        return None

    return matches.iloc[0]


# ============================================================
# BUILD DRIVER EVIDENCE
# ============================================================

def build_driver_evidence(row):

    evidence = {

        "driver": get_value(
            row,
            "driver"
        ),

        "driver_rank": safe_float(
            get_value(
                row,
                "hypothesis_rank"
            )
        ),

        "driver_change": safe_float(
            get_value(
                row,
                "driver_change"
            )
        ),

        # Analytical evidence
        "analytic_evidence_score": safe_float(
            get_value(
                row,
                "evidence_score"
            )
        ),

        "evidence_strength": get_value(
            row,
            "evidence_strength"
        ),

        "same_day_correlation": safe_float(
            get_value(
                row,
                "correlation_same_day"
            )
        ),

        "best_lag_days": safe_float(
            get_value(
                row,
                "best_lag_days"
            )
        ),

        "best_lag_correlation": safe_float(
            get_value(
                row,
                "best_lag_correlation"
            )
        ),

        "segment_consistency": safe_float(
            get_value(
                row,
                "segment_consistency"
            )
        ),

        "direction_alignment": get_value(
            row,
            "direction_alignment"
        ),

        # Historical evidence
        "recurrence_rate": safe_float(
            get_value(
                row,
                "recurrence_rate"
            )
        ),

        "lag_consistency": safe_float(
            get_value(
                row,
                "lag_consistency"
            )
        ),

        "direction_consistency": safe_float(
            get_value(
                row,
                "direction_consistency"
            )
        ),

        "comparable_event_count": safe_float(
            get_value(
                row,
                "comparable_event_count"
            )
        ),

        "historical_best_lag": safe_float(
            get_value(
                row,
                "historical_best_lag"
            )
        ),

        "historical_support_score": safe_float(
            get_value(
                row,
                "historical_support_score"
            )
        ),

        "validated_evidence_strength": get_value(
            row,
            "validated_evidence_strength"
        ),

        "hypothesis_status": get_value(
            row,
            "hypothesis_status"
        ),

        # Final ranking
        "final_hypothesis_score": safe_float(
            get_value(
                row,
                "final_hypothesis_score"
            )
        ),

        "ranking_strength": get_value(
            row,
            "ranking_strength"
        ),

        "investigation_status": get_value(
            row,
            "investigation_status"
        ),
    }

    return evidence


# ============================================================
# BUILD EVENT EVIDENCE
# ============================================================

def build_event_bundle(
    event,
    ranked_event
):

    date = event["date"]
    region = event["region"]

    # --------------------------------------------------------
    # Event information
    # --------------------------------------------------------

    event_bundle = {

        "event": {

            "date": str(date),

            "region": str(region),

            "metric": "revenue",

            "revenue": safe_float(
                get_value(
                    event,
                    "revenue"
                )
            ),

            "revenue_pct_change": safe_float(
                get_value(
                    event,
                    "revenue_pct_change_1d"
                )
            ),

            "revenue_z_score": safe_float(
                get_value(
                    event,
                    "revenue_z_score"
                )
            ),

            "revenue_absolute_impact": safe_float(
                get_value(
                    event,
                    "revenue_absolute_impact"
                )
            ),

            "materiality_score": safe_float(
                get_value(
                    event,
                    "materiality_score"
                )
            ),

            "priority": get_value(
                event,
                "priority"
            ),
        },

        "hypotheses": []
    }

    # --------------------------------------------------------
    # Hypotheses
    # --------------------------------------------------------

    if ranked_event is not None:

        ranked_event = ranked_event.sort_values(
            "hypothesis_rank"
        )

        for _, row in ranked_event.iterrows():

            hypothesis = build_driver_evidence(
                row
            )

            event_bundle["hypotheses"].append(
                hypothesis
            )

    return event_bundle


# ============================================================
# BUILD DATASET
# ============================================================

def build_evidence_bundle(
    materiality,
    ranked
):

    print("\nBuilding evidence bundles...")

    records = []

    # Normalize dates
    materiality = normalize_keys(
        materiality
    )

    ranked = normalize_keys(
        ranked
    )

    # --------------------------------------------------------
    # Each material event gets one evidence bundle
    # --------------------------------------------------------

    for _, event in materiality.iterrows():

        date = event["date"]
        region = event["region"]

        ranked_event = ranked[
            (ranked["date"] == date)
            &
            (ranked["region"] == region)
        ]

        bundle = build_event_bundle(
            event,
            ranked_event
        )

        # ----------------------------------------------------
        # Store JSON representation
        #
        # Parquet cannot directly store nested dictionaries
        # reliably, so we serialize the evidence bundle.
        # ----------------------------------------------------

        records.append({

            "date": date,

            "region": region,

            "metric": "revenue",

            "priority": get_value(
                event,
                "priority"
            ),

            "materiality_score": safe_float(
                get_value(
                    event,
                    "materiality_score"
                )
            ),

            "revenue": safe_float(
                get_value(
                    event,
                    "revenue"
                )
            ),

            "revenue_pct_change": safe_float(
                get_value(
                    event,
                    "revenue_pct_change_1d"
                )
            ),

            "hypothesis_count": len(
                bundle["hypotheses"]
            ),

            "evidence_bundle": json.dumps(
                bundle,
                default=str
            )
        })

    return pd.DataFrame(records)


# ============================================================
# TOP HYPOTHESIS SUMMARY
# ============================================================

def add_top_hypothesis_columns(df):

    print("Extracting top hypotheses...")

    top_driver = []
    top_score = []
    top_strength = []

    for _, row in df.iterrows():

        try:

            bundle = json.loads(
                row["evidence_bundle"]
            )

            hypotheses = bundle.get(
                "hypotheses",
                []
            )

            if hypotheses:

                top = hypotheses[0]

                top_driver.append(
                    top.get("driver")
                )

                top_score.append(
                    top.get(
                        "final_hypothesis_score"
                    )
                )

                top_strength.append(
                    top.get(
                        "ranking_strength"
                    )
                )

            else:

                top_driver.append(None)
                top_score.append(None)
                top_strength.append(None)

        except Exception:

            top_driver.append(None)
            top_score.append(None)
            top_strength.append(None)

    df["top_driver"] = top_driver

    df["top_driver_score"] = top_score

    df["top_driver_strength"] = top_strength

    return df


# ============================================================
# SAVE
# ============================================================

def save_results(df):

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✓ Saved evidence bundles:"
        f"\n  {OUTPUT_FILE}"
        f"\n  Rows    : {len(df)}"
        f"\n  Columns : {len(df.columns)}"
    )


# ============================================================
# DISPLAY SAMPLE
# ============================================================

def print_sample(df):

    print("\n" + "=" * 70)
    print("EVIDENCE BUILDER SAMPLE")
    print("=" * 70)

    columns = [
        "date",
        "region",
        "priority",
        "materiality_score",
        "revenue_pct_change",
        "hypothesis_count",
        "top_driver",
        "top_driver_score",
        "top_driver_strength",
    ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    print(
        df[columns]
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EVIDENCE BUILDER")
    print("=" * 70)

    (
        materiality,
        driver_features,
        validated,
        ranked
    ) = load_data()

    evidence = build_evidence_bundle(
        materiality,
        ranked
    )

    evidence = add_top_hypothesis_columns(
        evidence
    )

    save_results(
        evidence
    )

    print_sample(
        evidence
    )

    print("\n" + "=" * 70)
    print("EVIDENCE BUILDING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()