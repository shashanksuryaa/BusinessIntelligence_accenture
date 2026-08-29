import os
import json
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PROCESSED = os.path.join(
    BASE_DIR,
    "Data_processed"
)

INPUT_FILE = os.path.join(
    DATA_PROCESSED,
    "evidence_bundle.parquet"
)

OUTPUT_FILE = os.path.join(
    DATA_PROCESSED,
    "business_narratives.parquet"
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_percent(value, decimals=1):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value * 100:.{decimals}f}%"


def format_currency(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"₹{value:,.2f}"


def direction_text(value):
    value = safe_float(value)

    if value is None:
        return "changed"

    if value > 0:
        return "increased"

    if value < 0:
        return "decreased"

    return "remained stable"


# ============================================================
# LOAD DATA
# ============================================================

def load_evidence():

    print("Loading evidence bundles...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Evidence bundle not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Evidence rows loaded: {len(df)}")

    return df


# ============================================================
# PARSE EVIDENCE
# ============================================================

def parse_bundle(value):

    if isinstance(value, dict):
        return value

    try:
        return json.loads(value)
    except Exception:
        return {
            "event": {},
            "hypotheses": []
        }


# ============================================================
# SELECT TOP HYPOTHESES
# ============================================================

def select_hypotheses(
    hypotheses,
    max_hypotheses=3
):

    if not hypotheses:
        return []

    valid = []

    for h in hypotheses:

        score = safe_float(
            h.get("final_hypothesis_score")
        )

        if score is None:
            score = safe_float(
                h.get("analytic_evidence_score"),
                0
            )

        h["_sort_score"] = score

        valid.append(h)

    valid.sort(
        key=lambda x: x["_sort_score"],
        reverse=True
    )

    return valid[:max_hypotheses]


# ============================================================
# EVIDENCE STRENGTH TEXT
# ============================================================

def evidence_phrase(hypothesis):

    strength = hypothesis.get(
        "validated_evidence_strength"
    )

    if not strength:
        strength = hypothesis.get(
            "evidence_strength"
        )

    if strength == "HIGH":
        return "strong historical and analytical support"

    if strength == "MODERATE":
        return "moderate analytical support"

    if strength == "LOW":
        return "limited supporting evidence"

    return "available supporting evidence"


# ============================================================
# BUILD DRIVER SENTENCE
# ============================================================

def build_driver_sentence(hypothesis):

    driver = hypothesis.get(
        "driver",
        "unknown driver"
    )

    change = safe_float(
        hypothesis.get("driver_change")
    )

    if change is None:

        direction = "changed"

        magnitude = ""

    else:

        direction = (
            "increased"
            if change > 0
            else "decreased"
            if change < 0
            else "remained stable"
        )

        magnitude = (
            f" by {abs(change) * 100:.1f}%"
            if change != 0
            else ""
        )

    support = evidence_phrase(
        hypothesis
    )

    score = safe_float(
        hypothesis.get(
            "historical_support_score"
        )
    )

    if score is not None:

        return (
            f"{driver.replace('_', ' ').title()} "
            f"{direction}{magnitude}, "
            f"with {support} "
            f"(historical support score: {score:.2f})."
        )

    return (
        f"{driver.replace('_', ' ').title()} "
        f"{direction}{magnitude}, "
        f"with {support}."
    )


# ============================================================
# BUILD EVIDENCE SENTENCE
# ============================================================

def build_evidence_sentence(hypothesis):

    driver = hypothesis.get(
        "driver",
        "This driver"
    )

    same_day = safe_float(
        hypothesis.get(
            "same_day_correlation"
        )
    )

    best_lag = safe_float(
        hypothesis.get(
            "best_lag_days"
        )
    )

    best_corr = safe_float(
        hypothesis.get(
            "best_lag_correlation"
        )
    )

    recurrence = safe_float(
        hypothesis.get(
            "recurrence_rate"
        )
    )

    parts = []

    if same_day is not None:

        parts.append(
            f"same-day correlation was {same_day:.2f}"
        )

    if best_lag is not None and best_corr is not None:

        if best_lag == 0:

            lag_text = "same day"

        elif best_lag == 1:

            lag_text = "1 day"

        else:

            lag_text = f"{int(best_lag)} days"

        parts.append(
            f"strongest observed correlation was "
            f"{best_corr:.2f} at a {lag_text} lag"
        )

    if recurrence is not None:

        parts.append(
            f"similar patterns recurred in "
            f"{recurrence * 100:.1f}% of comparable events"
        )

    if not parts:

        return (
            f"Evidence was identified for "
            f"{driver.replace('_', ' ')}."
        )

    return (
        f"For {driver.replace('_', ' ')}, "
        + ", ".join(parts)
        + "."
    )


# ============================================================
# BUILD CONFIDENCE
# ============================================================

def determine_confidence(
    hypotheses
):

    if not hypotheses:
        return "LOW"

    top = hypotheses[0]

    validated_strength = top.get(
        "validated_evidence_strength"
    )

    ranking_strength = top.get(
        "ranking_strength"
    )

    if validated_strength == "HIGH":
        return "HIGH"

    if ranking_strength == "STRONG":
        return "HIGH"

    if validated_strength == "MODERATE":
        return "MEDIUM"

    return "LOW"


# ============================================================
# BUILD CAVEAT
# ============================================================

def build_caveat(
    hypotheses
):

    if not hypotheses:

        return (
            "Insufficient evidence was available "
            "to identify a reliable driver."
        )

    top = hypotheses[0]

    alignment = top.get(
        "direction_alignment"
    )

    consistency = safe_float(
        top.get(
            "direction_consistency"
        )
    )

    if (
        alignment == "INCONSISTENT_WITH_HISTORY"
        or (
            consistency is not None
            and consistency < 0.5
        )
    ):

        return (
            "The observed relationship is "
            "inconsistent with historical patterns. "
            "This should be investigated further "
            "before treating the driver as reliable."
        )

    return (
        "These findings indicate statistical and "
        "historical support, but do not establish "
        "causal certainty."
    )


# ============================================================
# BUILD NARRATIVE
# ============================================================

def generate_narrative(row):

    bundle = parse_bundle(
        row["evidence_bundle"]
    )

    event = bundle.get(
        "event",
        {}
    )

    hypotheses = bundle.get(
        "hypotheses",
        []
    )

    hypotheses = select_hypotheses(
        hypotheses
    )

    date = event.get(
        "date",
        str(row.get("date"))
    )

    region = event.get(
        "region",
        row.get("region")
    )

    revenue = safe_float(
        event.get("revenue")
    )

    revenue_change = safe_float(
        event.get("revenue_pct_change")
    )

    priority = event.get(
        "priority",
        row.get("priority")
    )

    confidence = determine_confidence(
        hypotheses
    )

    # --------------------------------------------------------
    # HEADLINE
    # --------------------------------------------------------

    if revenue_change is not None:

        if revenue_change < 0:

            headline = (
                f"Revenue declined by "
                f"{abs(revenue_change) * 100:.1f}% "
                f"in {region} on {date}."
            )

        elif revenue_change > 0:

            headline = (
                f"Revenue increased by "
                f"{revenue_change * 100:.1f}% "
                f"in {region} on {date}."
            )

        else:

            headline = (
                f"Revenue remained stable "
                f"in {region} on {date}."
            )

    else:

        headline = (
            f"Material revenue movement detected "
            f"in {region} on {date}."
        )

    # --------------------------------------------------------
    # EVENT DETAILS
    # --------------------------------------------------------

    if revenue is not None:

        event_sentence = (
            f"Revenue for the region was "
            f"{format_currency(revenue)}."
        )

    else:

        event_sentence = ""

    # --------------------------------------------------------
    # DRIVER SECTION
    # --------------------------------------------------------

    if hypotheses:

        primary_driver = hypotheses[0].get(
            "driver",
            "unknown"
        )

        driver_summary = (
            f"The strongest candidate driver is "
            f"{primary_driver.replace('_', ' ').title()}."
        )

        driver_sentences = []

        for hypothesis in hypotheses:

            driver_sentences.append(
                build_driver_sentence(
                    hypothesis
                )
            )

        evidence_sentences = []

        for hypothesis in hypotheses:

            evidence_sentences.append(
                build_evidence_sentence(
                    hypothesis
                )
            )

    else:

        primary_driver = None

        driver_summary = (
            "No sufficiently supported driver "
            "was identified."
        )

        driver_sentences = []

        evidence_sentences = []

    # --------------------------------------------------------
    # CAVEAT
    # --------------------------------------------------------

    caveat = build_caveat(
        hypotheses
    )

    # --------------------------------------------------------
    # FINAL NARRATIVE
    # --------------------------------------------------------

    sections = [

        headline,

        event_sentence,

        driver_summary,

        " ".join(driver_sentences),

        " ".join(evidence_sentences),

        f"Confidence: {confidence}.",

        caveat
    ]

    narrative = " ".join(
        s for s in sections
        if s
    )

    return {

        "date": date,

        "region": region,

        "metric": "revenue",

        "priority": priority,

        "materiality_score": safe_float(
            event.get(
                "materiality_score"
            )
        ),

        "revenue": revenue,

        "revenue_pct_change": revenue_change,

        "primary_driver": primary_driver,

        "confidence": confidence,

        "n_hypotheses": len(
            hypotheses
        ),

        "narrative": narrative,

        "caveat": caveat
    }


# ============================================================
# GENERATE ALL NARRATIVES
# ============================================================

def generate_all_narratives(df):

    print("\nGenerating business narratives...")

    records = []

    for _, row in df.iterrows():

        try:

            result = generate_narrative(
                row
            )

            records.append(result)

        except Exception as e:

            print(
                f"WARNING: Failed for "
                f"{row.get('date')} / "
                f"{row.get('region')}: {e}"
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# SAVE
# ============================================================

def save_results(df):

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n✓ Saved business narratives:"
        f"\n  {OUTPUT_FILE}"
        f"\n  Rows    : {len(df)}"
        f"\n  Columns : {len(df.columns)}"
    )


# ============================================================
# DISPLAY SAMPLE
# ============================================================

def print_sample(df):

    print("\n")
    print("=" * 70)
    print("NARRATIVE GENERATOR SAMPLE")
    print("=" * 70)

    columns = [
        "date",
        "region",
        "priority",
        "primary_driver",
        "confidence",
        "n_hypotheses",
        "narrative"
    ]

    print(
        df[columns]
        .head(5)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NARRATIVE GENERATOR")
    print("=" * 70)

    evidence = load_evidence()

    narratives = generate_all_narratives(
        evidence
    )

    save_results(
        narratives
    )

    print_sample(
        narratives
    )

    print("\n" + "=" * 70)
    print("NARRATIVE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()