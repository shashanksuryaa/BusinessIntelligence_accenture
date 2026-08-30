import os
import json
import re
import ast
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/workspaces/BusinessIntelligence_accenture")

EVIDENCE_PATH = (
    BASE_DIR / "Data_processed" / "evidence_bundle.parquet"
)

OUTPUT_PATH = (
    BASE_DIR / "Data_processed" / "orchestrated_insights.parquet"
)

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"


# ============================================================
# CONFIGURATION THRESHOLDS
# ============================================================

HIGH_DRIVER_SCORE = 0.70
MEDIUM_DRIVER_SCORE = 0.45

HIGH_HISTORICAL_SUPPORT = 0.70
MEDIUM_HISTORICAL_SUPPORT = 0.45

CONTRADICTION_CORRELATION_THRESHOLD = 0.20


# ============================================================
# RAG KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():
    """
    Loads markdown/text documents from knowledge_base/.

    These documents provide business context only.
    They are NOT treated as quantitative evidence.
    """

    documents = []

    if not KNOWLEDGE_BASE_DIR.exists():
        print("INFO: No local knowledge base found.")
        print(f"Expected directory: {KNOWLEDGE_BASE_DIR}")
        return documents

    for path in KNOWLEDGE_BASE_DIR.glob("*"):

        if path.suffix.lower() not in [".md", ".txt"]:
            continue

        try:
            text = path.read_text(
                encoding="utf-8"
            )

            documents.append(
                {
                    "source": path.name,
                    "text": text
                }
            )

        except Exception as exc:
            print(
                f"WARNING: Could not read {path.name}: {exc}"
            )

    return documents


def retrieve_context(driver, documents, top_k=3):
    """
    Lightweight keyword-based RAG retrieval.

    The RAG layer supplies business context.
    It does NOT override analytical evidence.
    """

    if not documents:
        return []

    driver = str(driver).lower()

    scored = []

    for doc in documents:

        text = doc["text"].lower()

        score = 0

        # Exact driver occurrence
        score += text.count(driver) * 3

        # Related business terms
        related_terms = {
            "orders": [
                "orders",
                "demand",
                "conversion",
                "inventory"
            ],
            "conversion_rate": [
                "conversion",
                "checkout",
                "campaign",
                "traffic"
            ],
            "marketing_spend": [
                "marketing",
                "campaign",
                "targeting",
                "demand"
            ],
            "aov": [
                "aov",
                "price",
                "pricing",
                "basket"
            ],
            "inventory_availability": [
                "inventory",
                "availability",
                "stock",
                "product"
            ],
            "delivery_delay_rate": [
                "delivery",
                "delay",
                "logistics",
                "operations"
            ],
            "return_rate": [
                "return",
                "refund",
                "product"
            ],
            "discount_rate": [
                "discount",
                "promotion",
                "pricing"
            ]
        }

        for term in related_terms.get(driver, []):
            score += text.count(term)

        scored.append(
            (
                score,
                doc
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for score, doc in scored[:top_k]
        if score > 0
    ]


# ============================================================
# JSON / EVIDENCE PARSING
# ============================================================

def parse_json_like(value):
    """
    Safely parse evidence_bundle values.

    Handles:
    - JSON strings
    - Python-style dictionaries/lists
    - already parsed objects
    """

    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, float) and pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    # Normal JSON
    try:
        return json.loads(value)
    except Exception:
        pass

    # Python representation
    try:
        return ast.literal_eval(value)
    except Exception:
        pass

    # Sometimes wrapped in quotes
    try:
        cleaned = value.strip('"')
        return json.loads(cleaned)
    except Exception:
        pass

    return None


def extract_driver_records(raw_bundle):
    """
    Extract driver-level evidence from evidence_bundle.

    Expected structure is generally something like:

    {
        "drivers": [
            {
                "driver": "orders",
                "analytic_evidence_score": ...,
                "historical_support_score": ...,
                "validated_evidence_strength": ...,
                ...
            }
        ]
    }

    Also supports nested lists/dictionaries.
    """

    parsed = parse_json_like(raw_bundle)

    if parsed is None:
        return []

    records = []

    def recursive_extract(obj):

        if isinstance(obj, dict):

            # Driver-level record
            if "driver" in obj:

                records.append(obj)

            for value in obj.values():
                recursive_extract(value)

        elif isinstance(obj, list):

            for item in obj:
                recursive_extract(item)

    recursive_extract(parsed)

    # Remove duplicates
    unique = []

    seen = set()

    for record in records:

        driver = str(
            record.get("driver", "")
        ).strip()

        if not driver:
            continue

        key = (
            driver,
            str(
                record.get(
                    "driver_rank",
                    ""
                )
            )
        )

        if key not in seen:

            seen.add(key)
            unique.append(record)

    return unique


# ============================================================
# DRIVER VALIDATION
# ============================================================

def numeric(value, default=0.0):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def normalize_strength(value):

    if value is None:
        return "UNKNOWN"

    value = str(value).upper().strip()

    if value in [
        "HIGH",
        "MEDIUM",
        "LOW"
    ]:
        return value

    return "UNKNOWN"


def validate_driver(record):
    """
    Determines whether the analytics pipeline actually
    validated the driver.

    IMPORTANT:
    The LLM/RAG layer does not perform statistics.
    It only interprets already-computed evidence.
    """

    driver = str(
        record.get(
            "driver",
            "unknown"
        )
    )

    analytic_score = numeric(
        record.get(
            "analytic_evidence_score"
        )
    )

    historical_score = numeric(
        record.get(
            "historical_support_score"
        )
    )

    final_score = numeric(
        record.get(
            "final_hypothesis_score"
        )
    )

    validated_strength = normalize_strength(
        record.get(
            "validated_evidence_strength"
        )
    )

    hypothesis_status = str(
        record.get(
            "hypothesis_status",
            ""
        )
    ).upper()

    investigation_status = str(
        record.get(
            "investigation_status",
            ""
        )
    ).upper()

    best_lag_corr = numeric(
        record.get(
            "best_lag_correlation"
        )
    )

    same_day_corr = numeric(
        record.get(
            "same_day_correlation"
        )
    )

    direction_alignment = str(
        record.get(
            "direction_alignment",
            ""
        )
    ).upper()

    # --------------------------------------------------------
    # Strong validation
    # --------------------------------------------------------

    strong_validation = (
        validated_strength == "HIGH"
        or
        (
            analytic_score >= HIGH_DRIVER_SCORE
            and historical_score >= HIGH_HISTORICAL_SUPPORT
        )
        or
        (
            final_score >= 0.70
            and historical_score >= 0.70
        )
    )

    # --------------------------------------------------------
    # Weak / contradictory evidence
    # --------------------------------------------------------

    weak_validation = (
        validated_strength == "LOW"
        or
        hypothesis_status in [
            "WEAK",
            "FAILED",
            "INVALID"
        ]
        or
        investigation_status == "DO_NOT_PRIORITIZE"
    )

    contradictory = False

    # Very weak analytical relationship
    if (
        abs(same_day_corr) < CONTRADICTION_CORRELATION_THRESHOLD
        and
        abs(best_lag_corr) < CONTRADICTION_CORRELATION_THRESHOLD
        and
        historical_score < MEDIUM_HISTORICAL_SUPPORT
    ):
        contradictory = True

    # Explicit direction conflict
    if "INCONSISTENT" in direction_alignment:
        contradictory = True

    # --------------------------------------------------------
    # Final classification
    # --------------------------------------------------------

    if strong_validation and not contradictory:

        return {
            "driver": driver,
            "validated": True,
            "confidence": "HIGH",
            "analytic_score": analytic_score,
            "historical_score": historical_score,
            "final_score": final_score,
            "reason": "Strong analytical and/or historical support."
        }

    if (
        not weak_validation
        and not contradictory
        and (
            analytic_score >= MEDIUM_DRIVER_SCORE
            or historical_score >= MEDIUM_HISTORICAL_SUPPORT
        )
    ):

        return {
            "driver": driver,
            "validated": True,
            "confidence": "MEDIUM",
            "analytic_score": analytic_score,
            "historical_score": historical_score,
            "final_score": final_score,
            "reason": "Moderate analytical or historical support."
        }

    return {
        "driver": driver,
        "validated": False,
        "confidence": "LOW",
        "analytic_score": analytic_score,
        "historical_score": historical_score,
        "final_score": final_score,
        "reason": (
            "Evidence is weak, contradictory, "
            "or insufficiently validated."
        )
    }


# ============================================================
# CONTRADICTION DETECTION
# ============================================================

def detect_contradiction(driver_records):

    if not driver_records:
        return False

    for record in driver_records:

        direction = str(
            record.get(
                "direction_alignment",
                ""
            )
        ).upper()

        if "INCONSISTENT" in direction:
            return True

        analytic = numeric(
            record.get(
                "analytic_evidence_score"
            )
        )

        historical = numeric(
            record.get(
                "historical_support_score"
            )
        )

        if (
            analytic >= 0.70
            and historical < 0.30
        ):
            return True

        if (
            historical >= 0.70
            and analytic < 0.30
        ):
            return True

    return False


# ============================================================
# RAG CONTEXT FORMATTER
# ============================================================

def format_context(context):

    if not context:
        return []

    formatted = []

    for doc in context:

        formatted.append(
            {
                "source": doc["source"],
                "text": doc["text"],
                "score": 1
            }
        )

    return formatted


# ============================================================
# DETERMINISTIC NARRATIVE
# ============================================================

def generate_supported_explanation(
    event,
    validated_driver,
    context
):

    date = event["date"]
    region = event["region"]

    revenue = numeric(
        event.get("revenue")
    )

    pct_change = numeric(
        event.get("revenue_pct_change")
    )

    driver = validated_driver["driver"]

    confidence = validated_driver["confidence"]

    direction = (
        "increased"
        if pct_change >= 0
        else "decreased"
    )

    return (
        f"Revenue {direction} by "
        f"{abs(pct_change):.1f}% in {region} "
        f"on {date}. Revenue was ₹{revenue:,.2f}. "
        f"The strongest supported driver is "
        f"{driver.replace('_', ' ').title()}, "
        f"with {confidence.lower()} confidence. "
        f"The evidence supports investigation and "
        f"interpretation but does not establish causality."
    )


def generate_abstention_explanation(
    event,
    candidate_driver,
    contradiction
):

    date = event["date"]
    region = event["region"]

    revenue = numeric(
        event.get("revenue")
    )

    pct_change = numeric(
        event.get("revenue_pct_change")
    )

    direction = (
        "increased"
        if pct_change >= 0
        else "decreased"
    )

    driver_text = (
        candidate_driver.replace(
            "_",
            " "
        ).title()
        if candidate_driver
        else "no definitive driver"
    )

    if contradiction:

        reason = (
            "Analytical and historical evidence "
            "is contradictory."
        )

    else:

        reason = (
            "Analytical and historical evidence "
            "is weak or insufficient."
        )

    return (
        f"Revenue {direction} by "
        f"{abs(pct_change):.1f}% in {region} "
        f"on {date}. Revenue was ₹{revenue:,.2f}. "
        f"The analytics pipeline identified "
        f"{driver_text} as a candidate driver, "
        f"but it was not sufficiently validated. "
        f"{reason} The system therefore abstains "
        f"from assigning a definitive driver and "
        f"recommends further investigation. "
        f"Event priority is {event['priority']}. "
        f"Observed analytical and historical "
        f"relationships do not establish causality."
    )


# ============================================================
# SINGLE EVENT ORCHESTRATION
# ============================================================

def orchestrate_event(
    event,
    knowledge_base
):

    raw_bundle = event.get(
        "evidence_bundle"
    )

    driver_records = extract_driver_records(
        raw_bundle
    )

    contradiction = detect_contradiction(
        driver_records
    )

    validated = []

    for record in driver_records:

        result = validate_driver(
            record
        )

        if result["validated"]:

            validated.append(
                result
            )

    # Sort strongest first
    validated.sort(
        key=lambda x: (
            x["final_score"],
            x["historical_score"],
            x["analytic_score"]
        ),
        reverse=True
    )

    candidate_driver = event.get(
        "top_driver"
    )

    if (
        candidate_driver is None
        or
        pd.isna(candidate_driver)
    ):
        candidate_driver = (
            validated[0]["driver"]
            if validated
            else None
        )

    # ========================================================
    # ABSTENTION
    # ========================================================

    if not validated or contradiction:

        context = retrieve_context(
            candidate_driver,
            knowledge_base
        )

        explanation = generate_abstention_explanation(
            event,
            candidate_driver,
            contradiction
        )

        return {
            "date": event["date"],
            "region": event["region"],
            "metric": event.get(
                "metric",
                "revenue"
            ),
            "event_status": "ABSTAIN",
            "confidence": "LOW",
            "primary_driver": candidate_driver,
            "primary_driver_score": numeric(
                event.get(
                    "top_driver_score"
                )
            ),
            "primary_driver_strength": event.get(
                "top_driver_strength"
            ),
            "priority": event["priority"],
            "materiality_score": numeric(
                event.get(
                    "materiality_score"
                )
            ),
            "revenue": numeric(
                event.get("revenue")
            ),
            "revenue_pct_change": numeric(
                event.get(
                    "revenue_pct_change"
                )
            ),
            "explanation": explanation,
            "retrieved_context": json.dumps(
                format_context(context),
                ensure_ascii=False
            ),
            "contradiction_detected": contradiction,
            "abstention_reason": (
                "CONTRADICTORY_EVIDENCE"
                if contradiction
                else "INSUFFICIENT_EVIDENCE"
            )
        }

    # ========================================================
    # SUPPORTED EVENT
    # ========================================================

    strongest = validated[0]

    context = retrieve_context(
        strongest["driver"],
        knowledge_base
    )

    explanation = generate_supported_explanation(
        event,
        strongest,
        context
    )

    return {
        "date": event["date"],
        "region": event["region"],
        "metric": event.get(
            "metric",
            "revenue"
        ),
        "event_status": "SUPPORTED",
        "confidence": strongest["confidence"],
        "primary_driver": strongest["driver"],
        "primary_driver_score": strongest[
            "final_score"
        ],
        "primary_driver_strength": strongest[
            "confidence"
        ],
        "priority": event["priority"],
        "materiality_score": numeric(
            event.get(
                "materiality_score"
            )
        ),
        "revenue": numeric(
            event.get("revenue")
        ),
        "revenue_pct_change": numeric(
            event.get(
                "revenue_pct_change"
            )
        ),
        "explanation": explanation,
        "retrieved_context": json.dumps(
            format_context(context),
            ensure_ascii=False
        ),
        "contradiction_detected": False,
        "abstention_reason": None
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LLM / RAG ORCHESTRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load evidence
    # --------------------------------------------------------

    print("\nLoading evidence bundle...")

    if not EVIDENCE_PATH.exists():

        raise FileNotFoundError(
            f"Evidence file not found:\n"
            f"{EVIDENCE_PATH}"
        )

    evidence = pd.read_parquet(
        EVIDENCE_PATH
    )

    print(
        f"Evidence rows loaded: "
        f"{len(evidence)}"
    )

    print(
        "Evidence columns:",
        evidence.columns.tolist()
    )

    required_columns = [
        "date",
        "region",
        "priority",
        "revenue",
        "revenue_pct_change",
        "evidence_bundle",
        "top_driver"
    ]

    missing = [
        col
        for col in required_columns
        if col not in evidence.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Date normalization
    # --------------------------------------------------------

    evidence["date"] = pd.to_datetime(
        evidence["date"]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    # --------------------------------------------------------
    # Load RAG
    # --------------------------------------------------------

    print("\nLoading RAG knowledge base...")

    knowledge_base = load_knowledge_base()

    print(
        f"Knowledge documents loaded: "
        f"{len(knowledge_base)}"
    )

    if knowledge_base:

        print(
            "Documents:",
            [
                x["source"]
                for x in knowledge_base
            ]
        )

    else:

        print("INFO: RAG context unavailable.")

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    print(
        f"\nEvents to process: "
        f"{len(evidence)}"
    )

    results = []

    for idx, (_, event) in enumerate(
        evidence.iterrows(),
        start=1
    ):

        print(
            f"Processing {idx}/{len(evidence)}..."
        )

        try:

            result = orchestrate_event(
                event,
                knowledge_base
            )

            results.append(
                result
            )

        except Exception as exc:

            print(
                f"WARNING: Orchestration failed "
                f"for {event.get('date')} / "
                f"{event.get('region')}: {exc}"
            )

            # Safe abstention
            results.append(
                {
                    "date": event.get(
                        "date"
                    ),
                    "region": event.get(
                        "region"
                    ),
                    "metric": event.get(
                        "metric",
                        "revenue"
                    ),
                    "event_status": "ABSTAIN",
                    "confidence": "LOW",
                    "primary_driver": event.get(
                        "top_driver"
                    ),
                    "primary_driver_score": numeric(
                        event.get(
                            "top_driver_score"
                        )
                    ),
                    "primary_driver_strength": event.get(
                        "top_driver_strength"
                    ),
                    "priority": event.get(
                        "priority"
                    ),
                    "materiality_score": numeric(
                        event.get(
                            "materiality_score"
                        )
                    ),
                    "revenue": numeric(
                        event.get(
                            "revenue"
                        )
                    ),
                    "revenue_pct_change": numeric(
                        event.get(
                            "revenue_pct_change"
                        )
                    ),
                    "explanation": (
                        "The orchestration layer "
                        "could not safely interpret "
                        "the supplied evidence. "
                        "The system abstains and "
                        "recommends investigation."
                    ),
                    "retrieved_context": "[]",
                    "contradiction_detected": False,
                    "abstention_reason": (
                        "ORCHESTRATION_ERROR"
                    )
                }
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = pd.DataFrame(
        results
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ORCHESTRATION COMPLETE")
    print("=" * 70)

    print("\n✓ Saved:")
    print(f"  {OUTPUT_PATH}")

    print(
        f"\nRows    : {len(output)}"
    )

    print(
        f"Columns : {len(output.columns)}"
    )

    print("\nStatus distribution:")

    print(
        output[
            "event_status"
        ].value_counts()
    )

    print("\nConfidence distribution:")

    print(
        output[
            "confidence"
        ].value_counts()
    )

    print("\nSample:")

    display_columns = [
        "date",
        "region",
        "event_status",
        "confidence",
        "primary_driver",
        "explanation"
    ]

    print(
        output[
            display_columns
        ].head(10).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()