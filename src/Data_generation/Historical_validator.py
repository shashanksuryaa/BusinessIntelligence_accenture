# src/Data_generation/historical_validator.py

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

ANALYSIS_FILE = os.path.join(
    DATA_DIR, "driver_analysis.parquet"
)

DRIVER_FEATURE_FILE = os.path.join(
    DATA_DIR, "driver_features.parquet"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR, "validated_hypotheses.parquet"
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

# A historical event must have a meaningful target movement.
# The materiality detector has already selected events, but
# this threshold is also used to find comparable past events.
TARGET_EVENT_THRESHOLD = 0.10

# Comparable historical events must be separated by this many
# days so that one long-running incident is not counted as many
# independent recurrences.
EVENT_SEPARATION_DAYS = 7

# How many days around a historical event we inspect for the
# candidate driver's movement.
LAG_WINDOWS = [0, 1, 3, 7]

# Minimum comparable events before giving strong historical
# support.
MIN_EVENTS_FOR_STRONG = 5
MIN_EVENTS_FOR_MODERATE = 3


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_mean(values):
    values = pd.Series(values).dropna()

    if len(values) == 0:
        return np.nan

    return values.mean()


def safe_rate(values):
    values = pd.Series(values).dropna()

    if len(values) == 0:
        return np.nan

    return values.mean()


def direction_sign(value, tolerance=1e-12):
    """
    Return -1, 0, +1 for a movement.
    """

    if pd.isna(value):
        return 0

    if abs(value) <= tolerance:
        return 0

    return int(np.sign(value))


def pct_change(current, previous):
    """
    Safe percentage change.
    """

    if (
        pd.isna(current)
        or pd.isna(previous)
        or previous == 0
    ):
        return np.nan

    return (current - previous) / abs(previous)


def historical_expected_direction(
    driver_change,
    historical_correlation,
):
    """
    Given a driver's historical relationship with revenue,
    determine the revenue direction expected from the driver's
    observed movement.

    This is NOT causal inference.
    """

    driver_sign = direction_sign(
        driver_change
    )

    corr_sign = direction_sign(
        historical_correlation
    )

    if driver_sign == 0 or corr_sign == 0:
        return 0

    return driver_sign * corr_sign


def direction_matches(
    revenue_change,
    driver_change,
    historical_correlation,
):
    """
    Check whether the historical relationship predicts the
    observed revenue movement.

    Example:
        correlation = -0.5
        driver increases
        revenue decreases

        => True
    """

    expected = historical_expected_direction(
        driver_change,
        historical_correlation,
    )

    actual = direction_sign(
        revenue_change
    )

    if expected == 0 or actual == 0:
        return np.nan

    return bool(
        expected == actual
    )


# ============================================================
# HISTORICAL EVENT DETECTION
# ============================================================

def detect_historical_events(
    df,
    exclude_material_event=None,
):
    """
    Detect historical revenue movements within each region.

    Events are based on the absolute 1-day revenue movement.
    Positive and negative movements are both retained.

    We deliberately keep the event definition simple and
    deterministic. The semantic/materiality layer determines
    which events deserve investigation; this function finds
    comparable historical movements.
    """

    working = df.sort_values(
        ["region", "date"]
    ).copy()

    if "revenue_pct_change_1d" not in working.columns:

        working["revenue_pct_change_1d"] = (
            working.groupby("region")["revenue"]
            .pct_change()
        )

    working["is_historical_event"] = (
        working["revenue_pct_change_1d"].abs()
        >= TARGET_EVENT_THRESHOLD
    )

    events = working[
        working["is_historical_event"]
    ].copy()

    # --------------------------------------------------------
    # Prevent one sustained movement from being counted as
    # several independent historical events.
    # --------------------------------------------------------

    selected = []

    for region, group in events.groupby(
        "region"
    ):

        group = group.sort_values(
            "date"
        )

        last_event_date = None

        for _, row in group.iterrows():

            current_date = pd.Timestamp(
                row["date"]
            )

            if (
                last_event_date is None
                or (
                    current_date
                    - last_event_date
                ).days
                >= EVENT_SEPARATION_DAYS
            ):

                selected.append(row)
                last_event_date = current_date

    events = pd.DataFrame(
        selected
    )

    if events.empty:
        return events

    events["date"] = pd.to_datetime(
        events["date"]
    )

    return events


# ============================================================
# FIND DRIVER MOVEMENT AROUND AN EVENT
# ============================================================

def get_driver_movement_at_lag(
    df,
    region,
    event_date,
    driver,
    lag,
):
    """
    Compare driver value at event_date-lag with the previous
    day at event_date-lag-1.

    This gives a clean movement measure:

        driver(t-lag) vs driver(t-lag-1)

    For lag 0:
        driver on event day vs previous day.

    For lag 3:
        driver three days before event vs four days before event.

    This lets us test whether a candidate driver repeatedly
    moves before/around a target event.
    """

    region_df = df[
        df["region"] == region
    ].sort_values("date")

    target_date = (
        pd.Timestamp(event_date)
        - pd.Timedelta(days=lag)
    )

    previous_date = (
        target_date
        - pd.Timedelta(days=1)
    )

    current_rows = region_df[
        region_df["date"] == target_date
    ]

    previous_rows = region_df[
        region_df["date"] == previous_date
    ]

    if (
        current_rows.empty
        or previous_rows.empty
    ):
        return np.nan

    current_value = current_rows.iloc[0].get(
        driver,
        np.nan,
    )

    previous_value = previous_rows.iloc[0].get(
        driver,
        np.nan,
    )

    return pct_change(
        current_value,
        previous_value,
    )


# ============================================================
# VALIDATE ONE DRIVER FOR ONE CURRENT EVENT
# ============================================================

def validate_driver(
    current_event,
    driver,
    analysis_row,
    historical_events,
    df,
):
    """
    Validate one candidate driver against comparable
    historical target movements.

    We compare historical events with the SAME revenue
    direction as the current event.

    Metrics:
        recurrence_rate
        lag_consistency
        direction_consistency
        comparable_event_count
        historical_driver_movement
        evidence score

    Again: this is historical observational validation,
    not causal inference.
    """

    current_date = pd.Timestamp(
        current_event["date"]
    )

    region = current_event["region"]

    current_revenue_change = (
        current_event.get(
            "revenue_pct_change_1d",
            np.nan,
        )
    )

    historical_correlation = (
        analysis_row.get(
            "correlation_same_day",
            np.nan,
        )
    )

    current_best_lag = analysis_row.get(
        "best_lag_days",
        np.nan,
    )

    # --------------------------------------------------------
    # Historical events from the same region
    # --------------------------------------------------------

    comparable = historical_events[
        historical_events["region"] == region
    ].copy()

    # Remove the current event itself.
    comparable = comparable[
        comparable["date"] != current_date
    ]

    # --------------------------------------------------------
    # Keep only events with the same target direction.
    #
    # A revenue decline is compared with prior declines,
    # while an increase is compared with prior increases.
    # --------------------------------------------------------

    current_target_sign = direction_sign(
        current_revenue_change
    )

    if current_target_sign != 0:

        comparable = comparable[
            comparable[
                "revenue_pct_change_1d"
            ].apply(direction_sign)
            == current_target_sign
        ]

    comparable_event_count = len(
        comparable
    )

    # --------------------------------------------------------
    # If no comparable history exists, return an explicit
    # insufficient-history result.
    # --------------------------------------------------------

    if comparable_event_count == 0:

        return {
            "recurrence_rate": np.nan,
            "lag_consistency": np.nan,
            "direction_consistency": np.nan,
            "comparable_event_count": 0,
            "historical_driver_event_count": 0,
            "historical_driver_mean_change": np.nan,
            "historical_driver_median_change": np.nan,
            "historical_best_lag": np.nan,
            "historical_best_lag_rate": np.nan,
            "historical_support_score": 0.0,
            "historical_support": "INSUFFICIENT_HISTORY",
        }

    # --------------------------------------------------------
    # Evaluate each historical event
    # --------------------------------------------------------

    recurrence_matches = []
    lag_matches = []
    direction_matches_list = []

    historical_changes = []

    # For each event, find the lag at which the driver movement
    # best matches the expected revenue direction.
    event_best_lags = []

    for _, historical_event in comparable.iterrows():

        historical_revenue_change = (
            historical_event[
                "revenue_pct_change_1d"
            ]
        )

        # --------------------------------------------
        # Expected direction at each lag
        # --------------------------------------------

        lag_results = []

        for lag in LAG_WINDOWS:

            driver_change = (
                get_driver_movement_at_lag(
                    df,
                    region,
                    historical_event["date"],
                    driver,
                    lag,
                )
            )

            if pd.isna(driver_change):
                continue

            expected_match = direction_matches(
                historical_revenue_change,
                driver_change,
                historical_correlation,
            )

            if pd.notna(expected_match):

                lag_results.append({
                    "lag": lag,
                    "driver_change": driver_change,
                    "match": bool(
                        expected_match
                    ),
                })

        if not lag_results:
            continue

        # --------------------------------------------
        # Does the driver support this event at ANY
        # of the tested lags?
        # --------------------------------------------

        event_has_match = any(
            x["match"]
            for x in lag_results
        )

        recurrence_matches.append(
            int(event_has_match)
        )

        # --------------------------------------------
        # Determine the best matching lag for this
        # historical event.
        # --------------------------------------------

        matching_lags = [
            x for x in lag_results
            if x["match"]
        ]

        if matching_lags:

            # Prefer the smallest lag because an earlier
            # signal is more useful operationally.
            best_event_lag = min(
                matching_lags,
                key=lambda x: x["lag"]
            )

            event_best_lags.append(
                best_event_lag["lag"]
            )

        # --------------------------------------------
        # Same-day direction consistency
        # --------------------------------------------

        same_day = next(
            (
                x
                for x in lag_results
                if x["lag"] == 0
            ),
            None,
        )

        if same_day is not None:

            direction_matches_list.append(
                int(same_day["match"])
            )

        # --------------------------------------------
        # Driver movement at the current candidate
        # lag
        # --------------------------------------------

        selected_lag = (
            int(current_best_lag)
            if pd.notna(current_best_lag)
            else 0
        )

        selected = next(
            (
                x
                for x in lag_results
                if x["lag"] == selected_lag
            ),
            None,
        )

        if selected is not None:

            historical_changes.append(
                selected["driver_change"]
            )

    # --------------------------------------------------------
    # Aggregate historical evidence
    # --------------------------------------------------------

    recurrence_rate = (
        safe_rate(
            recurrence_matches
        )
        if recurrence_matches
        else np.nan
    )

    direction_consistency = (
        safe_rate(
            direction_matches_list
        )
        if direction_matches_list
        else np.nan
    )

    historical_driver_mean_change = (
        safe_mean(
            historical_changes
        )
    )

    historical_driver_median_change = (
        pd.Series(
            historical_changes
        ).median()
        if historical_changes
        else np.nan
    )

    # --------------------------------------------------------
    # Lag consistency
    #
    # We calculate the share of historical events whose
    # strongest matching lag equals the current event's
    # candidate lag.
    # --------------------------------------------------------

    if event_best_lags:

        selected_lag = (
            int(current_best_lag)
            if pd.notna(current_best_lag)
            else 0
        )

        lag_consistency = np.mean(
            np.array(event_best_lags)
            == selected_lag
        )

        # Also report which lag occurred most frequently.
        lag_counts = pd.Series(
            event_best_lags
        ).value_counts()

        historical_best_lag = int(
            lag_counts.index[0]
        )

        historical_best_lag_rate = (
            lag_counts.iloc[0]
            / len(event_best_lags)
        )

    else:

        lag_consistency = np.nan
        historical_best_lag = np.nan
        historical_best_lag_rate = np.nan

    # --------------------------------------------------------
    # Evidence score
    # --------------------------------------------------------

    components = []

    if pd.notna(recurrence_rate):
        components.append(
            0.40 * recurrence_rate
        )

    if pd.notna(lag_consistency):
        components.append(
            0.25 * lag_consistency
        )

    if pd.notna(direction_consistency):
        components.append(
            0.20 * direction_consistency
        )

    # A small contribution from the analytical score helps
    # retain the candidate-generation information without
    # allowing it to dominate historical validation.
    analytical_score = analysis_row.get(
        "evidence_score",
        0.0,
    )

    if pd.notna(analytical_score):
        components.append(
            0.15 * float(
                np.clip(
                    analytical_score,
                    0,
                    1,
                )
            )
        )

    if components:
        historical_support_score = (
            sum(components)
            / (
                0.40
                + (
                    0.25
                    if pd.notna(
                        lag_consistency
                    )
                    else 0
                )
                + (
                    0.20
                    if pd.notna(
                        direction_consistency
                    )
                    else 0
                )
                + (
                    0.15
                    if pd.notna(
                        analytical_score
                    )
                    else 0
                )
            )
        )
    else:
        historical_support_score = 0.0

    # --------------------------------------------------------
    # Confidence/support bucket
    #
    # Strong support requires both a high score AND enough
    # historical events. This prevents a 100% recurrence on
    # only one or two examples from becoming "strong".
    # --------------------------------------------------------

    if (
        comparable_event_count >= MIN_EVENTS_FOR_STRONG
        and historical_support_score >= 0.70
    ):

        historical_support = "HIGH_HISTORICAL_SUPPORT"

    elif (
        comparable_event_count >= MIN_EVENTS_FOR_MODERATE
        and historical_support_score >= 0.45
    ):

        historical_support = "MODERATE_HISTORICAL_SUPPORT"

    elif comparable_event_count >= 1:

        historical_support = "WEAK_HISTORICAL_SUPPORT"

    else:

        historical_support = "INSUFFICIENT_HISTORY"

    return {
        "recurrence_rate": recurrence_rate,
        "lag_consistency": lag_consistency,
        "direction_consistency": direction_consistency,
        "comparable_event_count": comparable_event_count,
        "historical_driver_event_count": len(
            historical_changes
        ),
        "historical_driver_mean_change": (
            historical_driver_mean_change
        ),
        "historical_driver_median_change": (
            historical_driver_median_change
        ),
        "historical_best_lag": (
            historical_best_lag
        ),
        "historical_best_lag_rate": (
            historical_best_lag_rate
        ),
        "historical_support_score": (
            historical_support_score
        ),
        "historical_support": (
            historical_support
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HISTORICAL VALIDATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load analytical candidates
    # --------------------------------------------------------

    print(
        "\nLoading driver analysis..."
    )

    analysis = pd.read_parquet(
        ANALYSIS_FILE
    )

    analysis["date"] = pd.to_datetime(
        analysis["date"]
    )

    print(
        f"Analysis rows loaded: {len(analysis)}"
    )

    # --------------------------------------------------------
    # Load driver features
    # --------------------------------------------------------

    print(
        "\nLoading driver features..."
    )

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
    # Historical events
    # --------------------------------------------------------

    print(
        "\nDetecting historical revenue events..."
    )

    historical_events = (
        detect_historical_events(df)
    )

    print(
        f"Historical events found: "
        f"{len(historical_events)}"
    )

    # --------------------------------------------------------
    # Current material events are represented by the
    # analytical candidates.
    # --------------------------------------------------------

    current_events = (
        analysis[
            [
                "date",
                "region",
                "revenue_change",
            ]
        ]
        .drop_duplicates(
            ["date", "region"]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Validate every candidate driver.
    # --------------------------------------------------------

    print(
        "\nValidating historical recurrence..."
    )

    output_rows = []

    for _, event in current_events.iterrows():

        event_date = pd.Timestamp(
            event["date"]
        )

        region = event["region"]

        event_candidates = analysis[
            (analysis["date"] == event_date)
            & (analysis["region"] == region)
        ].copy()

        # Use the event's analytical revenue change.
        current_event = {
            "date": event_date,
            "region": region,
            "revenue_pct_change_1d": (
                event["revenue_change"]
            ),
        }

        for _, analysis_row in (
            event_candidates.iterrows()
        ):

            driver = analysis_row[
                "driver"
            ]

            validation = validate_driver(
                current_event=current_event,
                driver=driver,
                analysis_row=analysis_row,
                historical_events=historical_events,
                df=df,
            )

            row = analysis_row.to_dict()

            row.update(validation)

            output_rows.append(row)

    if not output_rows:

        print(
            "\nNo historical validation results generated."
        )

        return

    results = pd.DataFrame(
        output_rows
    )

    # --------------------------------------------------------
    # Final evidence classification
    # --------------------------------------------------------

    def combined_support(row):

        historical = row[
            "historical_support"
        ]

        analytical = row[
            "evidence_strength"
        ]

        if historical == "HIGH_HISTORICAL_SUPPORT":
            return "HIGH"

        if (
            historical
            == "MODERATE_HISTORICAL_SUPPORT"
            and analytical
            in ["STRONG", "MODERATE"]
        ):
            return "MODERATE"

        if historical == "INSUFFICIENT_HISTORY":
            return "INSUFFICIENT"

        return "LOW"

    results[
        "validated_evidence_strength"
    ] = results.apply(
        combined_support,
        axis=1,
    )

    # --------------------------------------------------------
    # Hypothesis status
    #
    # This is intentionally NOT "causal".
    # --------------------------------------------------------

    def hypothesis_status(row):

        if (
            row["validated_evidence_strength"]
            == "HIGH"
        ):
            return "SUPPORTED_FOR_INVESTIGATION"

        if (
            row["validated_evidence_strength"]
            == "MODERATE"
        ):
            return "PLAUSIBLE"

        if (
            row["validated_evidence_strength"]
            == "INSUFFICIENT"
        ):
            return "INSUFFICIENT_HISTORY"

        return "WEAK"

    results[
        "hypothesis_status"
    ] = results.apply(
        hypothesis_status,
        axis=1,
    )

    # --------------------------------------------------------
    # Rank after validation
    # --------------------------------------------------------

    results[
        "validated_rank"
    ] = (
        results
        .groupby(
            ["date", "region"]
        )[
            "historical_support_score"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
    )

    results = results.sort_values(
        [
            "date",
            "region",
            "validated_rank",
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
        "HISTORICAL VALIDATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nValidation rows : {len(results)}"
    )

    print(
        "\nValidated evidence distribution:"
    )

    print(
        results[
            "validated_evidence_strength"
        ].value_counts()
    )

    print(
        "\nHypothesis status:"
    )

    print(
        results[
            "hypothesis_status"
        ].value_counts()
    )

    print(
        "\nTop validated drivers:"
    )

    top = results[
        results["validated_rank"] == 1
    ][
        [
            "date",
            "region",
            "driver",
            "evidence_strength",
            "recurrence_rate",
            "lag_consistency",
            "direction_consistency",
            "comparable_event_count",
            "historical_best_lag",
            "historical_support_score",
            "validated_evidence_strength",
            "hypothesis_status",
        ]
    ].head(30)

    print(
        top.to_string(
            index=False
        )
    )

    print(
        "\n✓ Saved validated hypotheses:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "HISTORICAL VALIDATOR COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()