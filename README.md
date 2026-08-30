# Kachow — KPI Intelligence-to-Action Engine

## Accenture AI Innovation Challenge 2026 — Round 2

**Team Kachow · IIT (BHU), Varanasi**

---
##Dependencies
pandas          → DataFrames, Parquet processing, KPI calculations
numpy           → numerical operations
pyarrow         → reading/writing .parquet files
scipy           → statistical calculations
scikit-learn    → ML/statistical utilities if used in the pipeline
openai          → optional LLM integration
python-dotenv   → loading API keys from .env

## Why we built this

A dashboard can tell a business that revenue dropped by 20%.

It cannot, by itself, answer the questions that come next:

- Is the movement actually important?
- What is most likely driving it?
- Has this pattern happened before?
- Can we trust the available evidence?
- What should the business do next?
- And, just as importantly, when should we **not** pretend to know?

That is the gap we wanted to solve.

We built **KPI Intelligence-to-Action Engine**, an AI analyst that moves from a changing KPI to a decision-ready answer while keeping quantitative reasoning outside the LLM.

The system is designed around a simple principle:

> **First establish what the evidence says. Then use AI to explain it and help act on it.**

---

# 1. What the system does

At a high level:

```text
Business Data
     ↓
KPI Computation
     ↓
Materiality Detection
     ↓
Driver Analysis
     ↓
Historical Validation
     ↓
Hypothesis Ranking
     ↓
Evidence Bundle
     ↓
LLM / RAG Orchestration
     ↓
Persona-specific Insight
     ↓
Action Recommendation
```

There is also an important second path:

```text
Evidence
   ↓
Conflicting / insufficient?
   ↓
ABSTAIN
   ↓
Explain uncertainty
   ↓
Recommend investigation
```

The goal is not to generate a story for every KPI movement.

The goal is to generate a story **only when the evidence deserves one**.

---

# 2. The core idea: deterministic first, LLM second

One of the most important decisions in our solution was to avoid making the LLM responsible for quantitative truth.

The analytical pipeline calculates and validates:

- KPI values
- percentage changes
- materiality
- correlations
- lag relationships
- segment consistency
- recurrence
- historical support
- driver rankings
- evidence strength

The LLM does **not** recompute these values.

Instead, once the evidence is ready, the LLM/RAG layer can:

- retrieve relevant business context
- connect the evidence to business meaning
- synthesize an explanation
- communicate uncertainty
- personalize the output

This separation makes the system easier to audit and safer to use.

---

# 3. Data and KPI layer

For the prototype, we created synthetic business data representing several business systems:

- Customers
- Sales Orders
- Deliveries
- Inventory
- Marketing
- Web Sessions

These sources have different business roles and grains, allowing us to model a more realistic BI environment.

From this data, the system computes KPIs including:

- Revenue
- Orders
- Average Order Value (AOV)
- Conversion Rate
- Marketing Spend
- Inventory Availability
- Delivery Delay Rate
- Return Rate
- Discount Rate

We also maintain KPI semantic information such as:

- definition
- formula
- business role
- likely drivers
- data source
- refresh cadence
- grain
- dimensions
- materiality threshold
- lineage
- access restrictions

This gives the analytical layer a governed understanding of what each KPI means.

---

# 4. Materiality: not every change deserves an explanation

A business KPI moves every day.

We therefore do not send every movement downstream.

The materiality layer asks:

> **Is this movement large enough, unusual enough, and financially meaningful enough to investigate?**

It produces a materiality score and event priority such as:

```text
CRITICAL
HIGH
MEDIUM
LOW
```

Only material events move into the deeper driver analysis.

This is important for both usability and LLM economics:

```text
All KPI observations
        ↓
Materiality screening
        ↓
Only meaningful events
        ↓
Deep analytics / LLM
```

We avoid spending compute or model calls explaining normal business noise.

---

# 5. Driver analysis

Once a material movement is identified, the system looks for possible explanatory drivers.

For each candidate driver, we examine evidence such as:

- same-day correlation
- lagged correlation
- segment consistency
- direction alignment
- recurrence rate
- lag consistency
- direction consistency
- number of comparable historical events
- historical support

For example, if revenue changes and orders move strongly with it, Orders may become a high-ranked candidate.

But we do **not** stop at correlation.

That is why historical validation comes next.

---

# 6. Historical validation

A relationship that appears strong today may not be reliable historically.

The historical validator asks:

> **When a similar business event happened in the past, did the same driver behave in a similar way?**

We therefore evaluate:

- recurrence
- lag consistency
- direction consistency
- comparable event count
- historical support

This allows the system to distinguish:

```text
Strong analytical evidence
        +
Strong historical support
        ↓
More trustworthy hypothesis
```

from:

```text
Strong analytical signal
        +
Weak / inconsistent history
        ↓
Low confidence / abstain
```

This is one of the key safeguards in the system.

---

# 7. Hypothesis ranking

Multiple drivers can move at the same time.

Instead of forcing one explanation immediately, the system ranks candidate hypotheses using the analytical and historical evidence already calculated.

The output captures:

- driver rank
- analytical evidence
- historical support
- final hypothesis score
- ranking strength
- investigation status

This gives downstream components a structured evidence hierarchy rather than an unstructured list of possible causes.

---

# 8. Evidence bundle

The **evidence bundle** is the contract between our analytical pipeline and the intelligence layer.

It stores the information needed to explain an event without going back to raw data.

For example:

```text
date
region
metric
priority
materiality_score
revenue
revenue_pct_change
top_driver
top_driver_score
top_driver_strength
evidence_bundle
```

Inside the evidence bundle, driver-level evidence includes fields such as:

- driver
- driver rank
- analytical evidence score
- evidence strength
- same-day correlation
- lag information
- segment consistency
- historical support
- hypothesis status
- final hypothesis score
- investigation status

This is what allows us to keep the LLM grounded.

---

# 9. LLM + RAG orchestration

The LLM is used **after** the evidence has been established.

The RAG layer retrieves business context from a local knowledge base containing documents such as:

```text
knowledge_base/
├── orders.md
├── conversion_rate.md
├── aov.md
├── marketing_spend.md
├── inventory_availability.md
├── delivery_delay_rate.md
├── return_rate.md
├── discount_rate.md
└── playbooks/
```

These documents explain things like:

- what a KPI means
- which business levers are relevant
- what should be investigated
- which actions are appropriate
- what not to infer from correlation

### Why RAG?

The analytical dataset tells us **what happened**.

The knowledge base helps answer **what that means in a business context**.

For example:

```text
Analytics:
Orders is the highest-ranked driver.

RAG:
Orders is a volume driver of revenue.
Relevant areas to investigate include
conversion, marketing, inventory and checkout.

LLM:
Combines the evidence and business context
into a concise explanation.
```

The RAG layer therefore enriches the explanation without changing the underlying numbers.

---

# 10. The important part: knowing when to stop

A major requirement of our solution is uncertainty handling.

We explicitly designed the system so that it can say:

> **"We don't have enough evidence to give you a reliable answer."**

There are two important cases.

### Supported evidence

When the selected driver has sufficiently strong analytical and historical support:

```text
SUPPORTED
HIGH confidence
```

The system can continue to narrative and action generation.

### Low-confidence / contradictory evidence

When the evidence is weak or conflicting:

```text
ABSTAIN
LOW confidence
```

The system does not force a driver.

For example:

```text
Analytics:
Orders looks strongly associated with revenue.

Historical validation:
The same relationship is weak or inconsistent.

Decision:
Do not promote Orders as a definitive driver.
```

Instead, the system explains the uncertainty and recommends further investigation.

This prevents a high correlation from becoming an unsupported causal story.

---

# 11. Persona Engine

The same evidence is not equally useful to every person.

We therefore created a deterministic persona layer for:

### Executive

Focuses on:

- what changed
- business impact
- strongest supported driver
- priority
- concise decision-oriented interpretation

### Marketing

Focuses on:

- conversion
- marketing spend
- orders
- discounts
- demand-generation signals
- campaign investigation

### Operations

Focuses on:

- orders
- inventory
- delivery performance
- returns
- operational constraints

The underlying analytical evidence remains the same.

Only the **emphasis and communication** change.

---

# 12. Action Recommender

Once the evidence is sufficiently supported, the action layer converts it into a practical business recommendation.

The structure follows:

```text
Driver
   ↓
Controllable Lever
   ↓
Action
   ↓
Expected Objective
   ↓
Owner
   ↓
Confidence
```

Examples of action types:

```text
IMMEDIATE_ACTION
INVESTIGATION
MONITORING
```

A strong example might look like:

```text
Primary driver:
Orders

Action:
Launch a targeted sales promotion to support order volume
and improve conversion.

Owner:
Revenue Operations / Sales & Marketing

Confidence:
HIGH
```

Weak evidence should not automatically produce aggressive action. In those cases, the system should recommend investigation or monitoring.

---

# 13. Example end-to-end behavior

## Supported case

```text
Revenue movement detected
        ↓
Orders ranked as strongest driver
        ↓
Historical evidence supports relationship
        ↓
SUPPORTED / HIGH
        ↓
RAG retrieves business context
        ↓
Narrative generated
        ↓
Business action recommended
```

## Contradictory case

```text
Revenue movement detected
        ↓
Orders ranked as candidate driver
        ↓
Historical evidence conflicts
        ↓
ABSTAIN / LOW
        ↓
No definitive driver presented
        ↓
Further investigation recommended
```

This distinction is central to the system.

---

# 14. Current prototype outputs

The project produces structured Parquet artifacts at each major stage.

### Evidence

```text
Data_processed/evidence_bundle.parquet
```

### Orchestration

```text
Data_processed/orchestrated_insights.parquet
```

### Persona output

```text
Data_processed/persona_insights.parquet
```

### Actions

```text
Data_processed/action_recommendations.parquet
```

These intermediate files make the system traceable and easy to inspect.

---

# 15. Project structure

```text
BusinessIntelligence_accenture/
│
├── Data/
│   ├── raw/
│   └── processed/
│
├── Data_processed/
│   ├── evidence_bundle.parquet
│   ├── orchestrated_insights.parquet
│   ├── persona_insights.parquet
│   └── action_recommendations.parquet
│
├── knowledge_base/
│   ├── orders.md
│   ├── conversion_rate.md
│   ├── aov.md
│   ├── marketing_spend.md
│   ├── inventory_availability.md
│   ├── delivery_delay_rate.md
│   ├── return_rate.md
│   ├── discount_rate.md
│   └── playbooks/
│
├── src/
│   └── Data_generation/
│       ├── generate_data.py
│       ├── inspect_kpis.py
│       ├── driver_feature.py
│       ├── materiality_detector.py
│       ├── analytic_agent.py
│       ├── historical_validator.py
│       ├── hypothesis_ranking.py
│       ├── evidence_builder.py
│       ├── narrative_generator.py
│       ├── action_recommender.py
│       ├── persona_engine.py
│       ├── llm_orchestration.py
│       └── pipeline.py
│
└── README.md
```

---

# 16. LLM vs non-LLM: our explicit split

| Layer | Approach | Why |
|---|---|---|
| Data generation | Python / Pandas | Reproducible synthetic business data |
| KPI calculation | Deterministic logic | Quantitative correctness |
| Materiality | Statistics + business rules | Identify meaningful events |
| Driver analysis | Statistics | Measure relationships |
| Historical validation | Statistical analysis | Validate recurrence |
| Hypothesis ranking | Deterministic scoring | Transparent prioritization |
| Evidence bundle | Structured data | Traceability |
| RAG | Retrieval | Business context |
| LLM orchestration | LLM | Synthesis and contextual interpretation |
| Persona layer | Rules / structured prompting | Role-specific communication |
| Action recommender | Rules + LLM-ready interface | Convert evidence into actions |
| Abstention | Deterministic gate | Prevent unsupported explanations |

---

# 17. Cost and latency philosophy

The system is designed to avoid wasting model calls.

We first filter:

```text
All KPI observations
       ↓
Materiality
       ↓
Only important events
       ↓
Evidence validation
       ↓
Only useful evidence
       ↓
LLM / RAG
```

This reduces unnecessary:

- model calls
- tokens
- latency
- cost

Additional production optimizations could include:

- caching
- batching
- model routing
- retrieval limits
- token budgets
- cost-per-insight monitoring

The architecture is therefore designed to scale more sensibly than an LLM-first approach.

---

# 18. Feedback and future learning

The next natural extension is a human feedback loop.

An analyst or business user could mark an insight as:

- correct
- incorrect
- useful
- not useful
- action accepted
- action rejected

The feedback can then be used to improve:

- driver ranking
- confidence thresholds
- retrieval quality
- recommendations
- persona-specific outputs

The important governance principle remains unchanged:

> Feedback can improve the system, but it should not override governed KPI definitions or quantitative evidence.

---

# 19. Security and governance

The prototype includes the foundations for governed BI through:

- KPI semantic metadata
- lineage
- structured evidence
- persona-aware output
- access-control metadata
- separation of analytics and generation

A production implementation can extend this to:

- row-level security
- column-level security
- domain-level permissions
- enterprise identity
- sensitive-data controls
- audit logs

---

# 20. How this maps to the Round 2 objective

| Round 2 objective | Our approach |
|---|---|
| Detect and prioritise material KPI movements | Materiality engine |
| Reconcile heterogeneous information | Structured KPI layer + RAG |
| Identify and rank drivers | Analytics + hypothesis ranking |
| Generate persona-specific narratives | Persona Engine |
| Communicate uncertainty | Confidence scoring |
| Abstain when evidence is insufficient | Contradiction / abstention logic |
| Recommend practical actions | Action Recommender |
| Learn from feedback | Feedback-loop design |
| Operate under cost / latency constraints | Early filtering + controlled LLM use |
| Keep LLM separate from quantitative truth | Deterministic-first architecture |

---

# 21. What we would take into production next

The prototype gives us the core intelligence loop.

A production version would add:

- real enterprise data connectors
- stronger semantic governance
- vector-based retrieval
- enterprise authentication and authorization
- richer unstructured sources such as tickets and sales notes
- model-call telemetry
- token and cost tracking
- automated evaluation
- stronger causal inference where justified
- analyst feedback and learning
- monitoring for data and model drift

---

# 22. Final takeaway

We did not build another dashboard.

We built a **translation layer between changing business numbers and business decisions**.

The system asks:

```text
What changed?
     ↓
Is it actually material?
     ↓
What could explain it?
     ↓
Is that explanation supported historically?
     ↓
What business context is relevant?
     ↓
How confident are we?
     ↓
Should we explain it or abstain?
     ↓
What should this particular user do next?
```

The most important behavior is not that the system always gives an answer.

It is that the system can recognize when the evidence is not strong enough to deserve one.

> **Evidence first. Context second. AI where it adds value. Abstention when it does not.**
