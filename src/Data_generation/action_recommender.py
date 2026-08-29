import os
import json
import time
import pandas as pd

from huggingface_hub import InferenceClient


# ============================================================
# PATH CONFIGURATION
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
    "action_recommendations.parquet"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# You can change this through the environment:
#
# export HF_MODEL="Qwen/Qwen2.5-7B-Instruct"
#
# Default model below is a good starting point for this task.

MODEL = os.getenv(
    "HF_MODEL",
    "openai/gpt-oss-120b"
)

HF_TOKEN = os.getenv(
    "HF_TOKEN"
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

# Start small while testing.
# After confirming the output works, change this to None.

TEST_LIMIT = 10


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an Action Recommendation Agent within a Business Intelligence
decision-support system.

Your responsibility is to translate validated analytical evidence into
practical, evidence-grounded business recommendations.

The upstream analytical pipeline has already performed:

- KPI analysis
- Materiality detection
- Statistical deviation analysis
- Correlation analysis
- Lag analysis
- Regional/segment consistency analysis
- Historical validation
- Hypothesis ranking
- Evidence construction

You MUST NOT redo these calculations.

============================================================
EVIDENCE RULES
============================================================

1. Treat the supplied evidence as the ONLY source of factual information.

2. Do not invent metrics, percentages, drivers, events, historical patterns,
   or business facts that are not present in the supplied evidence.

3. Do not introduce a driver that is not present in the evidence.

4. Do not claim that a driver CAUSED the KPI movement.

5. Statistical correlation, lag relationships, recurrence, and historical
   validation provide supporting evidence but DO NOT establish causality.

6. Clearly distinguish between:
   - observed facts
   - statistically supported relationships
   - historical support
   - recommended actions

7. If evidence is weak, inconsistent, or insufficient, prefer INVESTIGATION
   or MONITORING rather than an aggressive intervention.

8. Use the strongest validated drivers when forming recommendations.

9. Preserve the event priority supplied by the analytical pipeline.
   Do not independently change CRITICAL, HIGH, MEDIUM, or LOW.

============================================================
ACTION SELECTION
============================================================

Choose the action type based on the evidence:

IMMEDIATE_ACTION:
Use when the event is materially important and the evidence provides
strong and consistent support for a practical intervention.

INVESTIGATION:
Use when a material event requires further investigation before taking
a potentially costly or irreversible action, especially when evidence is
moderate, weak, conflicting, or causal interpretation is uncertain.

MONITORING:
Use when the movement is relatively weak, uncertain, inconsistent, or
does not justify immediate intervention.

Recommendations must be:

- practical
- specific
- business-oriented
- proportional to the evidence
- concise
- directly connected to the identified drivers

Do not recommend unrelated actions.

============================================================
CONFIDENCE
============================================================

Confidence represents confidence in the RECOMMENDATION, not certainty
about causality.

Use:

HIGH:
Strong analytical and historical evidence with consistent direction.

MEDIUM:
Moderate evidence or some uncertainty remains.

LOW:
Weak, conflicting, or insufficient evidence.

Never use HIGH confidence to imply causal certainty.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY a single valid JSON object.

Do not return Markdown.

Do not use code fences.

Do not include explanations before or after the JSON.

Do not include comments inside the JSON.

Use double quotes for all JSON keys and string values.

Do not add fields that are not specified below.

The response MUST follow exactly this structure:

{
    "primary_action": "...",
    "secondary_actions": [
        "...",
        "..."
    ],
    "action_type": "IMMEDIATE_ACTION | INVESTIGATION | MONITORING",
    "recommended_owner": "...",
    "priority": "CRITICAL | HIGH | MEDIUM | LOW",
    "reason": "...",
    "expected_objective": "...",
    "confidence": "HIGH | MEDIUM | LOW",
    "causal_caveat": "..."
}

The "secondary_actions" field must always be a JSON array.
If there are no appropriate secondary actions, return an empty array.

The "priority" field must exactly match the priority supplied in the
input evidence.

The "causal_caveat" must explicitly acknowledge that the evidence does
not establish causality.
"""

# ============================================================
# LOAD EVIDENCE
# ============================================================

def load_evidence():

    print("Loading evidence bundles...")

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nEvidence bundle not found:\n{INPUT_FILE}"
        )

    evidence = pd.read_parquet(
        INPUT_FILE
    )

    print(
        f"Evidence rows loaded: {len(evidence)}"
    )

    print(
        f"Evidence columns: {len(evidence.columns)}"
    )

    return evidence


# ============================================================
# PARSE EVIDENCE BUNDLE
# ============================================================

def parse_bundle(value):

    # Sometimes parquet stores JSON as string.
    if isinstance(value, dict):
        return value

    if isinstance(value, str):

        try:

            return json.loads(value)

        except Exception:

            return {}

    return {}


# ============================================================
# EXTRACT HYPOTHESES
# ============================================================

def extract_hypotheses(bundle):

    hypotheses = bundle.get(
        "hypotheses",
        []
    )

    if not isinstance(
        hypotheses,
        list
    ):

        return []

    cleaned = []

    for hypothesis in hypotheses[:5]:

        if not isinstance(
            hypothesis,
            dict
        ):
            continue

        cleaned.append({

            "driver":
                hypothesis.get(
                    "driver"
                ),

            "driver_change":
                hypothesis.get(
                    "driver_change"
                ),

            "evidence_strength":
                hypothesis.get(
                    "evidence_strength"
                ),

            "validated_evidence_strength":
                hypothesis.get(
                    "validated_evidence_strength"
                ),

            "historical_support_score":
                hypothesis.get(
                    "historical_support_score"
                ),

            "recurrence_rate":
                hypothesis.get(
                    "recurrence_rate"
                ),

            "direction_consistency":
                hypothesis.get(
                    "direction_consistency"
                ),

            "best_lag_days":
                hypothesis.get(
                    "best_lag_days"
                ),

            "best_lag_correlation":
                hypothesis.get(
                    "best_lag_correlation"
                ),

            "direction_alignment":
                hypothesis.get(
                    "direction_alignment"
                )
        })

    return cleaned


# ============================================================
# PREPARE EVIDENCE FOR LLM
# ============================================================

def prepare_evidence(row):

    bundle = parse_bundle(
        row["evidence_bundle"]
    )

    event = bundle.get(
        "event",
        {}
    )

    hypotheses = extract_hypotheses(
        bundle
    )

    # If event information is absent from the bundle,
    # use the parquet row itself where possible.

    if not isinstance(
        event,
        dict
    ):

        event = {}

    event.setdefault(
        "date",
        str(row.get("date", ""))
    )

    event.setdefault(
        "region",
        row.get("region")
    )

    event.setdefault(
        "priority",
        row.get("priority", "MEDIUM")
    )

    event.setdefault(
        "materiality_score",
        row.get("materiality_score")
    )

    event.setdefault(
        "revenue",
        row.get("revenue")
    )

    event.setdefault(
        "revenue_pct_change",
        row.get("revenue_pct_change")
    )

    return {

        "event": event,

        "hypotheses": hypotheses
    }


# ============================================================
# BUILD LLM PROMPT
# ============================================================

def build_prompt(evidence):

    evidence_text = json.dumps(
        evidence,
        indent=2,
        default=str
    )

    return f"""
Analyze the following validated Business Intelligence evidence
and produce an evidence-grounded action recommendation.

VALIDATED EVIDENCE:
{evidence_text}

TASK:

1. Identify the strongest validated driver from the evidence.
2. Recommend the most appropriate practical business action.
3. Provide up to two relevant secondary actions.
4. Select exactly one action type:
   - IMMEDIATE_ACTION
   - INVESTIGATION
   - MONITORING
5. Select the most appropriate business owner or team.
6. Preserve the event priority exactly as provided.
7. Explain the recommendation using only the supplied evidence.
8. State the expected business objective.
9. Assign recommendation confidence based on the strength and consistency
   of the evidence.
10. Do not treat correlation or historical recurrence as proof of causality.

OUTPUT REQUIREMENTS:

Return ONLY a single valid JSON object.

Do not use Markdown.
Do not use code fences.
Do not include any text before or after the JSON.
Do not add fields outside the required schema.

Required schema:

{{
    "primary_action": "...",
    "secondary_actions": [
        "...",
        "..."
    ],
    "action_type": "IMMEDIATE_ACTION | INVESTIGATION | MONITORING",
    "recommended_owner": "...",
    "priority": "CRITICAL | HIGH | MEDIUM | LOW",
    "reason": "...",
    "expected_objective": "...",
    "confidence": "HIGH | MEDIUM | LOW",
    "causal_caveat": "..."
}}
"""


# ============================================================
# EXTRACT JSON FROM MODEL OUTPUT
# ============================================================

def extract_json(text):

    if not text:
        raise ValueError(
            "Empty model response."
        )

    text = text.strip()

    # Remove markdown fences if the model
    # accidentally returns them.

    if text.startswith("```"):

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()

    # First attempt: entire response.

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:
        pass

    # Second attempt:
    # extract first JSON object.

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        candidate = text[
            start:end + 1
        ]

        return json.loads(
            candidate
        )

    raise ValueError(
        "Could not extract valid JSON from model response."
    )


# ============================================================
# HUGGING FACE LLM CALL
# ============================================================

def generate_recommendation(
    client,
    evidence
):

    prompt = build_prompt(
        evidence
    )

    response = client.chat_completion(
        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        max_tokens=700
    )

    text = response.choices[0].message.content

    return extract_json(
        text
    )


# ============================================================
# FALLBACK RECOMMENDATION
# ============================================================

def fallback_recommendation(
    evidence
):

    event = evidence.get(
        "event",
        {}
    )

    hypotheses = evidence.get(
        "hypotheses",
        []
    )

    priority = event.get(
        "priority",
        "MEDIUM"
    )

    if hypotheses:

        strongest = hypotheses[0]

        driver = strongest.get(
            "driver",
            "identified driver"
        )

        driver_name = str(
            driver
        ).replace(
            "_",
            " "
        )

        primary_action = (
            f"Investigate the operational factors "
            f"behind the change in {driver_name}."
        )

    else:

        primary_action = (
            "Investigate the operational drivers "
            "behind the observed KPI movement."
        )

    return {

        "primary_action":
            primary_action,

        "secondary_actions": [

            "Review the affected region and relevant operational processes.",

            "Monitor the KPI and driver behavior for persistence."
        ],

        "action_type":
            "INVESTIGATION",

        "recommended_owner":
            "Business Analytics",

        "priority":
            priority,

        "reason":
            "The analytical pipeline identified a material "
            "KPI movement requiring further investigation.",

        "expected_objective":
            "Identify the operational factor responsible "
            "for the observed KPI movement.",

        "confidence":
            "LOW",

        "causal_caveat":
            "Statistical and historical evidence supports "
            "the hypothesis but does not establish causality."
    }


# ============================================================
# PROCESS ONE EVENT
# ============================================================

def process_event(
    client,
    row
):

    evidence = prepare_evidence(
        row
    )

    try:

        recommendation = generate_recommendation(
            client,
            evidence
        )

    except Exception as e:

        print(
            f"WARNING: LLM failed for "
            f"{row.get('date')} / "
            f"{row.get('region')}: {e}"
        )

        recommendation = fallback_recommendation(
            evidence
        )

    event = evidence["event"]

    return {

        "date":
            event.get("date"),

        "region":
            event.get("region"),

        "metric":
            event.get(
                "metric",
                "revenue"
            ),

        "revenue":
            event.get("revenue"),

        "revenue_pct_change":
            event.get(
                "revenue_pct_change"
            ),

        "materiality_score":
            event.get(
                "materiality_score"
            ),

        "priority":
            event.get(
                "priority"
            ),

        "primary_action":
            recommendation.get(
                "primary_action"
            ),

        "secondary_actions":
            json.dumps(
                recommendation.get(
                    "secondary_actions",
                    []
                ),
                ensure_ascii=False
            ),

        "action_type":
            recommendation.get(
                "action_type"
            ),

        "recommended_owner":
            recommendation.get(
                "recommended_owner"
            ),

        "reason":
            recommendation.get(
                "reason"
            ),

        "expected_objective":
            recommendation.get(
                "expected_objective"
            ),

        "confidence":
            recommendation.get(
                "confidence"
            ),

        "causal_caveat":
            recommendation.get(
                "causal_caveat"
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HUGGING FACE ACTION RECOMMENDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Check token
    # --------------------------------------------------------

    if not HF_TOKEN:

        raise RuntimeError(
            "\nHF_TOKEN is not set.\n"
            "Set it using:\n\n"
            'export HF_TOKEN="hf_xxxxxxxxx"\n'
        )

    print(
        f"\nModel: {MODEL}"
    )

    # --------------------------------------------------------
    # Create HF client
    # --------------------------------------------------------

    client = InferenceClient(
        model=MODEL,
        token=HF_TOKEN
    )

    # --------------------------------------------------------
    # Load evidence
    # --------------------------------------------------------

    evidence = load_evidence()

    # --------------------------------------------------------
    # Select rows
    # --------------------------------------------------------

    if TEST_LIMIT is None:

        rows_to_process = evidence

    else:

        rows_to_process = evidence.head(
            TEST_LIMIT
        )

    print(
        f"\nEvents to process: "
        f"{len(rows_to_process)}"
    )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    print(
        "\nGenerating action recommendations..."
    )

    results = []

    for i, (_, row) in enumerate(
        rows_to_process.iterrows(),
        start=1
    ):

        print(
            f"Processing "
            f"{i}/{len(rows_to_process)}..."
        )

        result = process_event(
            client,
            row
        )

        results.append(
            result
        )

        # Small delay to avoid hitting
        # provider rate limits.

        if i < len(rows_to_process):

            time.sleep(1)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = pd.DataFrame(
        results
    )

    output.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n✓ Saved action recommendations:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  Rows    : {len(output)}"
    )

    print(
        f"  Columns : {len(output.columns)}"
    )

    # --------------------------------------------------------
    # Display sample
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SAMPLE RECOMMENDATIONS"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "date",
        "region",
        "priority",
        "primary_action",
        "action_type",
        "recommended_owner",
        "confidence"
    ]

    print(
        output[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ACTION RECOMMENDER COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()