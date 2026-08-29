# Revenue Driver Rules

## Revenue Decomposition

Revenue can be interpreted using several business dimensions,
including:

- Order volume
- Average Order Value
- Conversion Rate
- Marketing activity
- Inventory availability
- Delivery performance
- Returns
- Discounts

## Evidence Priority

Quantitative analytical evidence is produced by the deterministic
analytics pipeline.

The RAG layer must not modify or override:
- KPI values
- Correlations
- Historical support scores
- Hypothesis scores
- Driver rankings
- Event priority
- Confidence scores

## Evidence Conflict

If analytical evidence and historical evidence disagree, the system
should communicate uncertainty rather than select a driver arbitrarily.

## Causality

Correlation, recurrence, historical association, or analytical
association does not establish causal certainty.

## Abstention

When evidence is insufficient or contradictory, the system should
abstain from identifying a definitive driver and recommend further
investigation.

## Action Principle

Recommended actions should follow:

driver
→ controllable lever
→ action
→ expected objective
→ owner
→ monitoring

## LLM Responsibility

The LLM may:
- synthesize evidence
- retrieve business context
- personalize explanations
- formulate natural-language recommendations

The LLM must not:
- calculate KPI values
- invent drivers
- override analytical rankings
- create unsupported quantitative claims
- claim causality