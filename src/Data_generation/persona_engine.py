"""
persona_engine.py

Generate persona-specific KPI insights from validated evidence.

INPUTS
------
Data_processed/evidence_bundle.parquet
Data_processed/materiality_events.parquet

OUTPUT
------
Data_processed/persona_insights.parquet

PERSONAS
--------
EXECUTIVE
OPERATIONS
MARKETING

IMPORTANT
---------
This module does NOT perform new statistical analysis.

All analytical evidence is taken from the upstream pipeline:
    - materiality detection
    - correlation analysis
    - lag analysis
    - segment consistency
    - historical validation
    - hypothesis ranking

The persona engine only interprets and presents that evidence
for different business personas.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "Data_processed"

EVIDENCE_FILE = DATA_DIR / "evidence_bundle.parquet"

MATERIALITY_FILE = DATA_DIR / "materiality_events.parquet"

OUTPUT_FILE = DATA_DIR / "persona_insights.parquet"


# ============================================================
# PERSONA CONFIGURATION
# ============================================================

PERSONA_CONFIG = {

    "EXECUTIVE": {
        "max_drivers": 3,
    },

    "OPERATIONS": {
        "max_drivers": 4,
    },

    "MARKETING": {
        "max_drivers": 4,
    },
}


# ============================================================
# DRIVER GROUPS
# ============================================================

MARKETING_DRIVERS = {
    "orders",
    "conversion_rate",
    "marketing_spend",
    "discount_rate",
    "aov",
}

OPERATIONS_DRIVERS = {
    "orders",
    "inventory_availability",
    "delivery_delay_rate",
    "return_rate",
    "aov",
    "conversion_rate",
}


# ============================================================
# EVIDENCE PRIORITY
# ============================================================

EVIDENCE_PRIORITY = {

    "STRONG": 1,
    "HIGH": 1,

    "MODERATE": 2,
    "MEDIUM": 2,

    "WEAK": 3,
    "LOW": 4,

}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=np.nan):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


def format_currency(value):

    value = safe_float(value)

    if pd.isna(value):

        return "N/A"

    return f"₹{value:,.2f}"


def format_driver_name(driver):

    if driver is None:
        return "Unknown driver"

    return str(driver).replace(
        "_",
        " "
    ).title()


def format_date(value):

    try:

        return pd.to_datetime(
            value
        ).strftime("%Y-%m-%d")

    except Exception:

        return str(value)


# ============================================================
# PARSE EVIDENCE BUNDLE
# ============================================================

def parse_evidence_bundle(raw_value):
    """
    Robustly extract driver-level evidence from evidence_bundle.

    Handles:
        - list of driver dictionaries
        - single driver dictionary
        - JSON string
        - Python dict/list string
        - nested {"drivers": [...]}
    """

    if raw_value is None:
        return []

    if isinstance(raw_value, float) and pd.isna(raw_value):
        return []

    # --------------------------------------------------------
    # Recursive parser
    # --------------------------------------------------------

    def extract(value):

        # Already a list
        if isinstance(value, list):

            drivers = []

            for item in value:

                if isinstance(item, dict):

                    # Nested drivers key
                    if "drivers" in item:

                        nested = extract(
                            item["drivers"]
                        )

                        drivers.extend(
                            nested
                        )

                    # Actual driver dictionary
                    elif "driver" in item:

                        drivers.append(item)

            return drivers

        # Dictionary
        if isinstance(value, dict):

            # Most likely structure:
            # {"drivers": [...]}
            if "drivers" in value:

                return extract(
                    value["drivers"]
                )

            # Single driver object
            if "driver" in value:

                return [value]

            # Sometimes evidence is nested
            # under another key
            drivers = []

            for nested_value in value.values():

                if isinstance(
                    nested_value,
                    (list, dict)
                ):

                    drivers.extend(
                        extract(
                            nested_value
                        )
                    )

            return drivers

        # String
        if isinstance(value, str):

            text = value.strip()

            if not text:
                return []

            # ----------------------------------------------
            # JSON
            # ----------------------------------------------

            try:

                parsed = json.loads(text)

                return extract(parsed)

            except Exception:
                pass

            # ----------------------------------------------
            # Python literal
            # ----------------------------------------------

            try:

                parsed = ast.literal_eval(text)

                return extract(parsed)

            except Exception:
                pass

        return []

    return extract(raw_value)
# ============================================================
# NORMALIZE EVIDENCE
# ============================================================

def normalize_evidence(evidence):

    rows = []

    for _, event in evidence.iterrows():

        raw_bundle = event.get(
            "evidence_bundle"
        )

        driver_list = parse_evidence_bundle(
            raw_bundle
        )
        if not driver_list:
            print("\nDEBUG: Could not parse evidence bundle")
            print("RAW TYPE:", type(raw_bundle))
            print("RAW VALUE:", raw_bundle)
        else:
            print(
                "DEBUG:",
                event.get("date"),
                event.get("region"),
                "drivers =",
                [
                   x.get("driver")
                   for x in driver_list
                ]
            )

        if not driver_list:
            continue

        for driver_info in driver_list:

            row = {

                # --------------------------------------------
                # Event information
                # --------------------------------------------

                "date": event.get("date"),

                "region": event.get("region"),

                "metric": event.get("metric"),

                "priority": event.get("priority"),

                "materiality_score":
                    event.get("materiality_score"),

                "revenue":
                    event.get("revenue"),

                "revenue_pct_change":
                    event.get("revenue_pct_change"),

                # --------------------------------------------
                # Event-level top driver
                # --------------------------------------------

                "top_driver":
                    event.get("top_driver"),

                "top_driver_score":
                    event.get("top_driver_score"),

                "top_driver_strength":
                    event.get("top_driver_strength"),

                # --------------------------------------------
                # Driver-level evidence
                # --------------------------------------------

                "driver":
                    driver_info.get("driver"),

                "driver_rank":
                    driver_info.get("driver_rank"),

                "driver_change":
                    driver_info.get("driver_change"),

                "analytic_evidence_score":
                    driver_info.get(
                        "analytic_evidence_score"
                    ),

                "evidence_strength":
                    driver_info.get(
                        "evidence_strength"
                    ),

                "same_day_correlation":
                    driver_info.get(
                        "same_day_correlation"
                    ),

                "best_lag_days":
                    driver_info.get(
                        "best_lag_days"
                    ),

                "best_lag_correlation":
                    driver_info.get(
                        "best_lag_correlation"
                    ),

                "segment_consistency":
                    driver_info.get(
                        "segment_consistency"
                    ),

                "direction_alignment":
                    driver_info.get(
                        "direction_alignment"
                    ),

                # --------------------------------------------
                # Historical validation
                # --------------------------------------------

                "recurrence_rate":
                    driver_info.get(
                        "recurrence_rate"
                    ),

                "lag_consistency":
                    driver_info.get(
                        "lag_consistency"
                    ),

                "direction_consistency":
                    driver_info.get(
                        "direction_consistency"
                    ),

                "comparable_event_count":
                    driver_info.get(
                        "comparable_event_count"
                    ),

                "historical_best_lag":
                    driver_info.get(
                        "historical_best_lag"
                    ),

                "historical_support_score":
                    driver_info.get(
                        "historical_support_score"
                    ),

                # --------------------------------------------
                # Hypothesis validation
                # --------------------------------------------

                "validated_evidence_strength":
                    driver_info.get(
                        "validated_evidence_strength"
                    ),

                "hypothesis_status":
                    driver_info.get(
                        "hypothesis_status"
                    ),

                "final_hypothesis_score":
                    driver_info.get(
                        "final_hypothesis_score"
                    ),

                "ranking_strength":
                    driver_info.get(
                        "ranking_strength"
                    ),

                "investigation_status":
                    driver_info.get(
                        "investigation_status"
                    ),
            }

            rows.append(row)

    if not rows:

        return pd.DataFrame()

    result = pd.DataFrame(
        rows
    )

    # ========================================================
    # DATE
    # ========================================================

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce"
    )

    # ========================================================
    # NUMERIC COLUMNS
    # ========================================================

    numeric_columns = [

        "materiality_score",

        "revenue",

        "revenue_pct_change",

        "top_driver_score",

        "driver_rank",

        "driver_change",

        "analytic_evidence_score",

        "same_day_correlation",

        "best_lag_days",

        "best_lag_correlation",

        "segment_consistency",

        "recurrence_rate",

        "lag_consistency",

        "direction_consistency",

        "comparable_event_count",

        "historical_best_lag",

        "historical_support_score",

        "final_hypothesis_score",

    ]

    for column in numeric_columns:

        if column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

    return result


# ============================================================
# RANK DRIVERS FOR PERSONA
# ============================================================

def rank_drivers(event_evidence, persona):

    data = event_evidence.copy()

    if data.empty:

        return data

    # --------------------------------------------------------
    # Persona-specific filtering
    # --------------------------------------------------------

    if persona == "MARKETING":

        filtered = data[
            data["driver"].isin(
                MARKETING_DRIVERS
            )
        ].copy()

        if filtered.empty:

            filtered = data.copy()

    elif persona == "OPERATIONS":

        filtered = data[
            data["driver"].isin(
                OPERATIONS_DRIVERS
            )
        ].copy()

        if filtered.empty:

            filtered = data.copy()

    else:

        filtered = data.copy()

    # --------------------------------------------------------
    # Clean evidence strength
    # --------------------------------------------------------

    filtered["evidence_priority"] = (
        filtered[
            "validated_evidence_strength"
        ]
        .map(EVIDENCE_PRIORITY)
        .fillna(
            filtered[
                "evidence_strength"
            ]
            .map(EVIDENCE_PRIORITY)
            .fillna(5)
        )
    )

    # --------------------------------------------------------
    # Numeric scores
    # --------------------------------------------------------

    filtered[
        "historical_support_score"
    ] = pd.to_numeric(
        filtered[
            "historical_support_score"
        ],
        errors="coerce"
    ).fillna(0)

    filtered[
        "final_hypothesis_score"
    ] = pd.to_numeric(
        filtered[
            "final_hypothesis_score"
        ],
        errors="coerce"
    ).fillna(0)

    filtered[
        "analytic_evidence_score"
    ] = pd.to_numeric(
        filtered[
            "analytic_evidence_score"
        ],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Ranking
    #
    # Prefer the upstream driver rank first.
    # Then use validated evidence.
    # --------------------------------------------------------

    filtered["driver_rank"] = pd.to_numeric(
        filtered["driver_rank"],
        errors="coerce"
    )

    filtered = filtered.sort_values(
        [
            "driver_rank",
            "evidence_priority",
            "historical_support_score",
            "final_hypothesis_score",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
        na_position="last",
    )

    return filtered.head(
        PERSONA_CONFIG[
            persona
        ]["max_drivers"]
    )


# ============================================================
# EXECUTIVE INSIGHT
# ============================================================

def build_executive_insight(
    event,
    drivers
):

    date = format_date(
        event["date"]
    )

    region = event["region"]

    revenue = event.get(
        "revenue"
    )

    priority = event.get(
        "priority",
        "LOW"
    )

    revenue_change = safe_float(
        event.get(
            "revenue_pct_change"
        )
    )

    # --------------------------------------------------------
    # Revenue movement
    # --------------------------------------------------------

    if pd.isna(revenue_change):

        change_text = (
            "A material revenue movement was detected"
        )

    elif revenue_change >= 0:

        change_text = (
            f"Revenue increased by "
            f"{revenue_change * 100:.1f}%"
        )

    else:

        change_text = (
            f"Revenue decreased by "
            f"{abs(revenue_change) * 100:.1f}%"
        )

    # --------------------------------------------------------
    # Strongest driver
    # --------------------------------------------------------

    if not drivers.empty:

        strongest = drivers.iloc[0]

        driver_name = format_driver_name(
            strongest.get("driver")
        )

        evidence_strength = (
            strongest.get(
                "validated_evidence_strength"
            )
        )

        if pd.isna(evidence_strength):

            evidence_strength = (
                strongest.get(
                    "evidence_strength",
                    "LOW"
                )
            )

        support = safe_float(
            strongest.get(
                "historical_support_score"
            )
        )

        if not pd.isna(support):

            driver_detail = (
                f"{driver_name} "
                f"(historical support "
                f"{support:.2f})"
            )

        else:

            driver_detail = driver_name

    else:

        driver_detail = (
            "No sufficiently validated driver"
        )

        evidence_strength = "LOW"

    # --------------------------------------------------------
    # Narrative
    # --------------------------------------------------------

    return (
        f"{change_text} in {region} on {date}. "
        f"Revenue was {format_currency(revenue)}. "
        f"The strongest validated driver is "
        f"{driver_detail}, with "
        f"{str(evidence_strength).lower()} evidence. "
        f"Event priority is {priority}. "
        f"The evidence supports investigation and "
        f"decision-making but does not establish causality."
    )


# ============================================================
# OPERATIONS INSIGHT
# ============================================================

def build_operations_insight(
    event,
    drivers
):

    date = format_date(
        event["date"]
    )

    region = event["region"]

    if drivers.empty:

        return (
            f"No sufficiently validated operational "
            f"drivers were identified for {region} "
            f"on {date}. Further investigation is recommended."
        )

    driver_descriptions = []

    for _, row in drivers.iterrows():

        driver_name = format_driver_name(
            row.get("driver")
        )

        support = safe_float(
            row.get(
                "historical_support_score"
            )
        )

        evidence_strength = row.get(
            "validated_evidence_strength"
        )

        if pd.isna(evidence_strength):

            evidence_strength = row.get(
                "evidence_strength"
            )

        if not pd.isna(support):

            description = (
                f"{driver_name} "
                f"(historical support "
                f"{support:.2f})"
            )

        else:

            description = driver_name

        if evidence_strength:

            description += (
                f" [{str(evidence_strength).lower()} evidence]"
            )

        driver_descriptions.append(
            description
        )

    return (
        f"Operational review for {region} "
        f"on {date} should focus on: "
        f"{', '.join(driver_descriptions)}. "
        f"These drivers have analytical and/or "
        f"historical support. The evidence indicates "
        f"where to investigate, but does not establish "
        f"causality."
    )


# ============================================================
# MARKETING INSIGHT
# ============================================================

def build_marketing_insight(
    event,
    drivers
):

    date = format_date(
        event["date"]
    )

    region = event["region"]

    if drivers.empty:

        return (
            f"No sufficiently validated marketing-related "
            f"drivers were identified for {region} "
            f"on {date}. Marketing investigation should "
            f"focus on available demand and conversion signals."
        )

    descriptions = []

    for _, row in drivers.iterrows():

        driver = row.get(
            "driver"
        )

        if driver not in MARKETING_DRIVERS:
            continue

        driver_name = format_driver_name(
            driver
        )

        support = safe_float(
            row.get(
                "historical_support_score"
            )
        )

        if not pd.isna(support):

            descriptions.append(
                f"{driver_name} "
                f"(historical support "
                f"{support:.2f})"
            )

        else:

            descriptions.append(
                driver_name
            )

    if not descriptions:

        return (
            f"No sufficiently validated marketing-related "
            f"drivers were identified for {region} "
            f"on {date}."
        )

    return (
        f"Marketing-related signals for {region} "
        f"on {date}: "
        f"{', '.join(descriptions)}. "
        f"These signals should guide campaign, "
        f"conversion and demand investigation. "
        f"Observed relationships do not establish causality."
    )


# ============================================================
# BUILD INSIGHT
# ============================================================

def build_insight(
    event,
    drivers,
    persona
):

    if persona == "EXECUTIVE":

        return build_executive_insight(
            event,
            drivers
        )

    if persona == "OPERATIONS":

        return build_operations_insight(
            event,
            drivers
        )

    if persona == "MARKETING":

        return build_marketing_insight(
            event,
            drivers
        )

    raise ValueError(
        f"Unknown persona: {persona}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PERSONA ENGINE")
    print("=" * 70)

    # ========================================================
    # LOAD EVIDENCE
    # ========================================================

    print()
    print("Loading evidence bundle...")

    if not EVIDENCE_FILE.exists():

        raise FileNotFoundError(
            f"Evidence bundle not found:\n"
            f"{EVIDENCE_FILE}"
        )

    evidence = pd.read_parquet(
        EVIDENCE_FILE
    )

    print(
        f"Evidence rows loaded: "
        f"{len(evidence)}"
    )

    print(
        f"Evidence columns: "
        f"{len(evidence.columns)}"
    )

    print(
        f"Evidence columns: "
        f"{list(evidence.columns)}"
    )

    # ========================================================
    # LOAD MATERIALITY EVENTS
    # ========================================================

    print()
    print("Loading materiality events...")

    if not MATERIALITY_FILE.exists():

        raise FileNotFoundError(
            f"Materiality file not found:\n"
            f"{MATERIALITY_FILE}"
        )

    materiality = pd.read_parquet(
        MATERIALITY_FILE
    )

    print(
        f"Material events loaded: "
        f"{len(materiality)}"
    )

    # ========================================================
    # DATE NORMALIZATION
    # ========================================================

    evidence["date"] = pd.to_datetime(
        evidence["date"],
        errors="coerce"
    )

    materiality["date"] = pd.to_datetime(
        materiality["date"],
        errors="coerce"
    )

    # ========================================================
    # NORMALIZE EVIDENCE BUNDLE
    # ========================================================

    print()
    print("Parsing driver-level evidence...")

    evidence = normalize_evidence(
        evidence
    )

    if evidence.empty:

        raise RuntimeError(
            "No driver-level evidence could be extracted "
            "from evidence_bundle.parquet."
        )

    print(
        f"Driver-level evidence rows: "
        f"{len(evidence)}"
    )

    print(
        f"Unique drivers: "
        f"{evidence['driver'].nunique()}"
    )

    # ========================================================
    # PREPARE EVENTS
    # ========================================================

    required_event_columns = [
        "date",
        "region",
        "priority",
        "revenue",
        "revenue_pct_change",
    ]

    available_columns = [
        column
        for column in required_event_columns
        if column in materiality.columns
    ]

    events = materiality[
        available_columns
    ].drop_duplicates(
        subset=[
            "date",
            "region",
        ]
    )

    print()
    print(
        f"Events to process: "
        f"{len(events)}"
    )

    # ========================================================
    # GENERATE PERSONA INSIGHTS
    # ========================================================

    results = []

    personas = [
        "EXECUTIVE",
        "OPERATIONS",
        "MARKETING",
    ]

    print()
    print(
        "Generating persona insights..."
    )

    for event_number, (_, event) in enumerate(
        events.iterrows(),
        start=1
    ):

        date = event["date"]

        region = event["region"]

        date_text = format_date(
            date
        )

        print(
            f"Processing {event_number}/"
            f"{len(events)}: "
            f"{date_text} / {region}"
        )

        # ----------------------------------------------------
        # Match evidence by event
        # ----------------------------------------------------

        event_evidence = evidence[
            (
                evidence["date"] == date
            )
            &
            (
                evidence["region"] == region
            )
        ].copy()

        if event_evidence.empty:

            print(
                f"WARNING: No evidence found for "
                f"{date_text} / {region}"
            )

            continue

        # ----------------------------------------------------
        # Generate each persona
        # ----------------------------------------------------

        for persona in personas:

            try:

                drivers = rank_drivers(
                    event_evidence,
                    persona
                )

                insight = build_insight(
                    event,
                    drivers,
                    persona
                )

                # --------------------------------------------
                # Driver names
                # --------------------------------------------

                top_drivers = []

                if not drivers.empty:

                    top_drivers = (
                        drivers[
                            "driver"
                        ]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )

                # --------------------------------------------
                # Strongest driver
                # --------------------------------------------

                if not drivers.empty:

                    strongest_driver = (
                        drivers.iloc[0]
                    )

                    strongest_driver_name = (
                        strongest_driver.get(
                            "driver"
                        )
                    )

                    strongest_support = safe_float(
                        strongest_driver.get(
                            "historical_support_score"
                        )
                    )

                    strongest_strength = (
                        strongest_driver.get(
                            "validated_evidence_strength"
                        )
                    )

                    if pd.isna(
                        strongest_strength
                    ):

                        strongest_strength = (
                            strongest_driver.get(
                                "evidence_strength",
                                "LOW"
                            )
                        )

                else:

                    strongest_driver_name = None

                    strongest_support = np.nan

                    strongest_strength = "LOW"

                # --------------------------------------------
                # Save result
                # --------------------------------------------

                results.append({

                    "date":
                        date_text,

                    "region":
                        region,

                    "persona":
                        persona,

                    "priority":
                        event.get(
                            "priority",
                            "LOW"
                        ),

                    "metric":
                        event.get(
                            "metric",
                            "revenue"
                        ),

                    "revenue":
                        event.get(
                            "revenue"
                        ),

                    "revenue_pct_change":
                        event.get(
                            "revenue_pct_change"
                        ),

                    "top_drivers":
                        ", ".join(
                            top_drivers
                        ),

                    "strongest_driver":
                        strongest_driver_name,

                    "strongest_driver_support":
                        strongest_support,

                    "evidence_strength":
                        strongest_strength,

                    "driver_count":
                        len(top_drivers),

                    "insight":
                        insight,

                    "causal_caveat":
                        (
                            "Analytical and historical "
                            "evidence does not establish "
                            "causality."
                        ),

                    "generated_by":
                        "Deterministic Persona Engine",

                })

            except Exception as exc:

                print(
                    f"WARNING: Persona processing failed "
                    f"for {date_text} / {region} / "
                    f"{persona}: {exc}"
                )

    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:

        raise RuntimeError(
            "No persona insights were generated."
        )

    # ========================================================
    # SORT
    # ========================================================

    result_df = result_df.sort_values(
        [
            "date",
            "region",
            "persona",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE
    # ========================================================

    result_df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # OUTPUT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PERSONA ENGINE COMPLETE")
    print("=" * 70)

    print()
    print("✓ Saved persona insights:")

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  Rows    : {len(result_df)}"
    )

    print(
        f"  Columns : {len(result_df.columns)}"
    )

    print()
    print("Persona distribution:")

    print(
        result_df[
            "persona"
        ].value_counts()
    )

    print()
    print("=" * 70)
    print("SAMPLE PERSONA INSIGHTS")
    print("=" * 70)

    print()

    print(
        result_df[
            [
                "date",
                "region",
                "persona",
                "priority",
                "top_drivers",
                "insight",
            ]
        ]
        .head(9)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()