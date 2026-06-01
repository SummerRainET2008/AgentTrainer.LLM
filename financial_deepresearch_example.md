# Finance DeepResearch Example

## Table of Contents

- [1. Input and output](#1-input-and-output)
- [2. Expert QA data and SOPs extraction](#2-expert-qa-data-and-sops-extraction)
  - [SOP1: Durable Pricing Power Evaluation](#sop1-durable-pricing-power-evaluation)
  - [SOP2: Revenue Quality Evaluation](#sop-2-revenue-quality-evaluation)
  - [SOP3: Customer Concentration Interpretation](#sop-3-customer-concentration-interpretation)
  - [SOP4: Platform vs Feature Evaluation](#sop-4-platform-vs-feature-evaluation)
  - [SOP5: Conditional Diligence Decision](#sop-5-conditional-diligence-decision)
- [3. New Query](#3-new-query)
  - [User Query](#user-query)
- [4. Query Understanding](#4-query-understanding)
- [5. Similar SOP Retrieval](#5-similar-sop-retrieval)
  - [Retrieved Primary SOP Candidates](#retrieved-primary-sop-candidates)
  - [Selected Primary SOP](#selected-primary-sop)
  - [Selected secondary SOPs](#selected-secondary-sops)
  - [Selected Sub-SOPs](#selected-sub-sops)
- [6. Compact SOP Bundle](#6-compact-sop-bundle)
- [7. Query Decomposition into Search Tasks](#7-query-decomposition-into-search-tasks)
- [8. Mock Evidence Found](#8-mock-evidence-found)
- [9. Structured Evidence Reorganization](#9-structured-evidence-reorganization)
- [10. Final Answer](#10-final-answer)
  - [Conclusion](#conclusion)
  - [Why the opportunity is interesting](#why-the-opportunity-is-interesting)
  - [Why the evidence is not yet enough](#why-the-evidence-is-not-yet-enough)
  - [Proceed conditions](#proceed-conditions)
  - [Pass conditions](#pass-conditions)
  - [Confidence](#confidence)

---
## 1. Input and output
Input: 
>expert QA data, propritary documents, a new user query.

Output:
>1. Extracted SOPs and sub-SOPs.
>1. Query analysis
>1. Sub-queries/sub-goals.
>1. Reorganized Structural evidence.
>1. Final answer

## 2. Expert QA data and SOPs extraction

### SOP1: Durable Pricing Power Evaluation

___Question___: 
>Does Company A, a vertical AI software startup for insurance claims processing, have durable pricing power, or is current margin expansion mostly driven by temporary AI enthusiasm?

___Expert Answer___: 
>Company A shows early signs of pricing power, but I would not yet call it durable. The strongest positive signal is >that the product is being used inside claims-adjustment workflows rather than as a generic AI assistant. If >adjusters depend on it for document review, fraud flagging, and claim summarization, switching cost may emerge from> workflow integration and historical claim data.
>
>However, usage intensity is not the same as pricing power. To prove pricing power, I would want evidence that >customers accepted price increases, renewed at higher ACV, expanded seat count, or reduced discounting. Current >evidence mainly supports adoption, not willingness to pay.
>
>The biggest risk is foundation-model commoditization. If Company A’s advantage is mostly prompt engineering or UI, >competitors can copy it. If it owns insurance-specific workflows, evaluation logic, integrations, and customer data> loops, durability is more credible.
>
>My conclusion is moderate near-term pricing power but unproven long-term durability. The next diligence should focus on renewal cohorts, price-increase acceptance, discounting, workflow dependency, and competitor substitution.
>
>Confidence: Medium.


#### Extract SOPs

```json
{
  "primary_sop": "Durable Pricing Power Evaluation",
  "intent": "investment_thesis_evaluation",
  "applies_to": [
    "pricing power",
    "margin durability",
    "willingness to pay",
    "durable monetization"
  ],
  "reasoning_steps": [
    "Separate near-term pricing power from durable pricing power",
    "Check whether adoption reflects real willingness to pay",
    "Evaluate workflow embeddedness and switching cost",
    "Check renewal, expansion, discounting, and price-increase evidence",
    "Assess commoditization or substitution risk",
    "State missing evidence and confidence"
  ],
  "required_evidence": [
    "price increase acceptance",
    "renewal ACV",
    "seat expansion",
    "discounting trend",
    "workflow integration",
    "customer interviews",
    "competitor alternatives"
  ],
  "failure_modes": [
    "Do not confuse high usage with pricing power",
    "Do not infer durable moat from temporary AI enthusiasm",
    "Do not use margin expansion alone as proof of pricing power"
  ]
}
```

#### Sub-SOPs

```json
[
  {
    "name": "Customer Willingness-to-Pay Check",
    "purpose": "Verify whether customers actually tolerate higher prices.",
    "required_evidence": [
      "price increase acceptance",
      "renewal at higher ACV",
      "seat expansion",
      "discounting trend"
    ]
  },
  {
    "name": "Workflow Embeddedness Check",
    "purpose": "Check whether the product is embedded in a mission-critical workflow.",
    "required_evidence": [
      "daily workflow usage",
      "system integrations",
      "process dependency",
      "migration friction"
    ]
  },
  {
    "name": "Commoditization Risk Check",
    "purpose": "Check whether competitors or foundation models can copy the product.",
    "required_evidence": [
      "competitor overlap",
      "foundation-model substitution",
      "unique data",
      "vertical workflow logic"
    ]
  }
]
```

---

### SOP 2: Revenue Quality Evaluation

___Question___: 
>Company B’s revenue is growing 90% YoY, but most growth comes from payment volume. Is this high-quality software revenue or lower-quality transaction-driven growth?

___Expert Answer___:
>I would not treat 90% revenue growth as automatically high quality. For a fintech infrastructure company, the key >question is whether revenue behaves like recurring software revenue or like transaction-volume-driven take-rate >revenue.
>
>First, separate subscription revenue from payment volume or take-rate revenue. Subscription revenue with high >retention usually has better predictability. Transaction revenue can grow fast but may be more exposed to volume >cyclicality, pricing compression, fraud loss, and compliance cost.
>
>Second, examine gross margin by revenue type. If payment volume grows but compliance, support, fraud, and payment >failure costs scale with volume, then operating leverage may be weaker than headline growth suggests.
>
>Third, check customer expansion. If customers are expanding because the product owns more workflow, that is >positive. If expansion is just higher payment throughput from a few customers, quality is weaker.
>
>My conclusion is that Company B’s revenue quality is mixed. Growth is impressive, but we need revenue composition, >gross margin by product line, take-rate trend, and customer-level expansion to judge whether this is software-like growth.
>
>Confidence: Medium-low until revenue breakdown is available.
  
#### Extract SOPs 

```json
{
  "primary_sop": "Revenue Quality Evaluation",
  "intent": "financial_quality_assessment",
  "applies_to": [
    "growth quality",
    "software revenue",
    "transaction revenue",
    "unit economics",
    "take-rate durability"
  ],
  "reasoning_steps": [
    "Do not judge quality from headline growth alone",
    "Separate software/subscription revenue from transaction/take-rate revenue",
    "Analyze gross margin by revenue type",
    "Check whether costs scale with volume",
    "Assess whether customer expansion reflects workflow ownership or volume growth",
    "Conclude with evidence gaps and confidence"
  ],
  "required_evidence": [
    "revenue breakdown",
    "subscription vs transaction revenue",
    "gross margin by product line",
    "take-rate trend",
    "fraud loss",
    "compliance cost",
    "support cost",
    "customer expansion"
  ],
  "failure_modes": [
    "Do not call transaction volume software revenue",
    "Do not infer operating leverage from top-line growth alone",
    "Do not ignore compliance and fraud costs"
  ]
}
```

#### Sub-SOPs

```json
[
  {
    "name": "Software vs Transaction Revenue Separation",
    "purpose": "Separate recurring software revenue from volume-driven transaction revenue.",
    "required_evidence": [
      "revenue composition",
      "subscription revenue",
      "payment volume",
      "take-rate revenue"
    ]
  },
  {
    "name": "Gross Margin Durability Check",
    "purpose": "Check whether margins improve with scale or are consumed by variable costs.",
    "required_evidence": [
      "gross margin by revenue type",
      "compliance cost",
      "fraud loss",
      "support burden"
    ]
  },
  {
    "name": "Expansion Quality Check",
    "purpose": "Distinguish workflow-driven expansion from pure volume growth.",
    "required_evidence": [
      "NRR",
      "account expansion reason",
      "workflow adoption",
      "payment throughput concentration"
    ]
  }
]
```

---

### SOP 3: Customer Concentration Interpretation

___Question___: 
>Company C has strong growth, but 48% of revenue comes from its top three customers. Should we treat this as validation or risk?

___Expert Answer___:
>Customer concentration can be either a validation signal or a risk signal depending on why it exists. I would not >automatically penalize the company, especially if it is early-stage, but I would require a much deeper >account-level analysis.
>
>If the top customers are strategic design partners, expanding across teams, adopting more modules, and renewing on >healthy terms, concentration can be a positive sign. It may indicate deep product-market fit with sophisticated >customers.
>
>But if the top customers require heavy customization, receive large discounts, have strong bargaining power, or >account for low gross margin revenue, then concentration is a major risk. The company may look larger than it >really is, while being fragile underneath.
>
>The key evidence is account expansion, renewal history, gross margin by customer, customization burden, contract >terms, and whether similar customers are converting.
>
>My conclusion is conditional: customer concentration is acceptable only if top accounts are expanding profitably and> repeatably. Otherwise, it is a serious revenue durability risk.
>
>Confidence: Medium.

#### Extract SOPs


```json
{
  "primary_sop": "Customer Concentration Interpretation",
  "intent": "risk_analysis",
  "applies_to": [
    "customer concentration",
    "top customer risk",
    "strategic account validation",
    "revenue durability"
  ],
  "reasoning_steps": [
    "Do not automatically classify concentration as good or bad",
    "Determine whether top customers are strategic validation or fragile dependency",
    "Check expansion, renewal, margin, customization, and bargaining power",
    "Assess whether similar customers are repeatably converting",
    "Classify concentration as acceptable, risky, or unresolved"
  ],
  "required_evidence": [
    "top customer revenue share",
    "account expansion",
    "renewal history",
    "gross margin by customer",
    "customization cost",
    "discounting",
    "contract terms",
    "pipeline of similar customers"
  ],
  "failure_modes": [
    "Do not treat large customers as validation without expansion evidence",
    "Do not ignore margin dilution from customization",
    "Do not ignore customer bargaining power"
  ]
}
```

#### Sub-SOPs

```json
[
  {
    "name": "Strategic Account Validation Check",
    "purpose": "Check whether top customers validate product-market fit.",
    "required_evidence": [
      "multi-team expansion",
      "module expansion",
      "renewal history",
      "reference quality"
    ]
  },
  {
    "name": "Fragile Dependency Check",
    "purpose": "Check whether top customers create revenue fragility.",
    "required_evidence": [
      "revenue concentration",
      "discounts",
      "customization burden",
      "low-margin customers",
      "termination risk"
    ]
  },
  {
    "name": "Repeatability Check",
    "purpose": "Check whether similar customers can be acquired repeatedly.",
    "required_evidence": [
      "conversion of similar customers",
      "sales cycle",
      "implementation burden",
      "pipeline quality"
    ]
  }
]
```

---

### SOP 4: Platform vs Feature Evaluation

___Question___:
> Is Company D becoming a workflow platform, or is it just a feature that ERP vendors can copy?

___Expert Answer___:
> The distinction between platform and feature depends on workflow ownership. A feature improves a specific step. A platform becomes the place where users repeatedly coordinate work, store operational history, manage exceptions, and make decisions.
>
>I would check where Company D sits in the customer workflow. If it only automates one task at the edge of the process, ERP vendors can likely copy or bundle it. But if Company D owns vendor onboarding, approval rules, audit history, reconciliation, exception handling, and cross-team collaboration, it is closer to a workflow platform.
>
>The second test is data persistence. If customers would lose important operational history, rules, templates, or >integrations by switching away, then platform potential is stronger.
>
>The third test is expansion path. A platform should naturally expand into adjacent workflows. A feature usually has >limited expansion surface.
>
>My conclusion: Company D is a platform candidate only if it owns persistent workflow data and multi-step operational processes. Otherwise, it is vulnerable to ERP bundling.
>
>Confidence: Medium.


#### Extract SOPs 
```json
{
  "primary_sop": "Platform vs Feature Evaluation",
  "intent": "moat_analysis",
  "applies_to": [
    "platform risk",
    "feature risk",
    "workflow ownership",
    "incumbent bundling"
  ],
  "reasoning_steps": [
    "Define whether the product owns a workflow or only improves one task",
    "Check depth of workflow ownership",
    "Check persistence of operational data, rules, templates, and integrations",
    "Assess expansion path into adjacent workflows",
    "Evaluate incumbent bundling risk",
    "Conclude platform, feature, or unresolved"
  ],
  "required_evidence": [
    "workflow map",
    "user behavior",
    "stored operational data",
    "system integrations",
    "approval rules",
    "audit history",
    "adjacent workflow expansion",
    "incumbent feature overlap"
  ],
  "failure_modes": [
    "Do not confuse productivity improvement with platform ownership",
    "Do not ignore incumbent bundling risk",
    "Do not call a single-point tool a platform without expansion evidence"
  ]
}
```

#### Sub-SOPs

```json
[
  {
    "name": "Workflow Ownership Evaluation",
    "purpose": "Determine whether the company owns a full workflow or only one feature.",
    "required_evidence": [
      "workflow map",
      "approval process",
      "collaboration behavior",
      "exception handling"
    ]
  },
  {
    "name": "Persistent Data Lock-in Check",
    "purpose": "Check whether stored data, rules, or integrations create switching cost.",
    "required_evidence": [
      "stored history",
      "templates",
      "rules",
      "audit trails",
      "integrations"
    ]
  },
  {
    "name": "Incumbent Bundling Risk Check",
    "purpose": "Assess whether ERP, bank, or platform incumbents can copy or bundle the feature.",
    "required_evidence": [
      "ERP overlap",
      "incumbent roadmap",
      "customer alternatives"
    ]
  }
]
```

---

## SOP 5: Conditional Diligence Decision

___Question___:
>Given mixed signals on growth, margin quality, and competition, should we proceed to deeper diligence on Company E or pass now?

___Expert Answer___:
>I would not make this a binary decision too early. When signals are mixed, the right answer is often to define >diligence gates. We should proceed only if the next diligence round can resolve the highest-impact uncertainties.
>
>The first step is to separate known positives from unresolved assumptions. Strong growth, enthusiastic customers, >and a large market may justify further work. But margin quality, competitive defensibility, and sales repeatability> may still be unproven.
>
>The second step is to identify what evidence would change our mind. For Company E, I would need to see cohort >expansion, revenue composition, customer-level margin, competitive win/loss data, and evidence of workflow lock-in.
>
>The third step is to define pass conditions. If growth is driven by low-margin custom work, if customers are not >repeatable, or if competitors can bundle the core feature, we should pass.
>
>My conclusion is proceed with targeted diligence, not full conviction. The next round should be designed around the key uncertainties, not general information gathering.


#### Extract SOPs 

```json
{
  "primary_sop": "Conditional Diligence Decision",
  "intent": "investment_decision_support",
  "applies_to": [
    "proceed or pass",
    "mixed signals",
    "diligence gates",
    "investment decision"
  ],
  "reasoning_steps": [
    "Avoid premature yes/no decision when signals are mixed",
    "Separate known positives from unresolved assumptions",
    "Identify evidence that would increase or decrease conviction",
    "Define proceed conditions and pass conditions",
    "Design next diligence around highest-impact uncertainties",
    "Give decision posture and confidence"
  ],
  "required_evidence": [
    "growth quality",
    "cohort expansion",
    "revenue composition",
    "customer-level margin",
    "competitive win/loss",
    "workflow lock-in",
    "repeatability"
  ],
  "failure_modes": [
    "Do not proceed with vague diligence",
    "Do not convert uncertainty into conviction",
    "Do not ignore pass conditions"
  ]
}
```

#### Sub-SOPs

```json
[
  {
    "name": "Known Positives vs Assumptions Separation",
    "purpose": "Separate confirmed evidence from unproven assumptions.",
    "required_evidence": [
      "confirmed positives",
      "unproven assumptions",
      "source confidence"
    ]
  },
  {
    "name": "Diligence Gate Definition",
    "purpose": "Define what must be true to proceed.",
    "required_evidence": [
      "must-prove conditions",
      "must-not-be-true conditions",
      "decision thresholds"
    ]
  },
  {
    "name": "Pass Condition Formulation",
    "purpose": "Define what evidence would justify passing.",
    "required_evidence": [
      "low-quality growth",
      "weak repeatability",
      "competitive substitutability",
      "margin weakness"
    ]
  }
]
```

---

## 2. New Query

### User Query

> Should we proceed with diligence on Company Z, a B2B fintech workflow platform for mid-market AP automation? Revenue grew 85% YoY, but 55% of revenue comes from the top three customers, a large portion of revenue is payment-volume driven, and ERP vendors are starting to bundle similar workflow features.

This query is deliberately complex because it touches multiple SOPs:

```text
diligence decision
+ revenue quality
+ customer concentration
+ platform vs feature
+ incumbent bundling risk
```

---

## 3. Query Understanding

```json
{
  "intent": "investment_decision_support",
  "entities": ["Company Z"],
  "sector": "B2B fintech workflow / AP automation",
  "business_model": [
    "workflow software",
    "payment-volume-driven revenue"
  ],
  "core_concepts": [
    "revenue quality",
    "customer concentration",
    "workflow ownership",
    "platform vs feature",
    "ERP bundling risk",
    "diligence decision"
  ],
  "implicit_dimensions": [
    "subscription vs transaction revenue",
    "gross margin durability",
    "top customer quality",
    "repeatability of sales",
    "switching cost",
    "incumbent competition"
  ],
  "output_format": "decision-oriented answer with proceed/pass conditions, evidence gaps, and confidence"
}
```

---

## 4. Similar SOP Retrieval

### Retrieved Primary SOP Candidates

```json
[
  {
    "sop": "Conditional Diligence Decision",
    "match_score": 0.94,
    "reason": "The query asks whether to proceed with diligence under mixed signals."
  },
  {
    "sop": "Revenue Quality Evaluation",
    "match_score": 0.91,
    "reason": "Revenue grew quickly but is partly payment-volume driven."
  },
  {
    "sop": "Customer Concentration Interpretation",
    "match_score": 0.89,
    "reason": "55% of revenue comes from top three customers."
  },
  {
    "sop": "Platform vs Feature Evaluation",
    "match_score": 0.86,
    "reason": "ERP vendors are bundling similar workflow features."
  },
  {
    "sop": "Durable Pricing Power Evaluation",
    "match_score": 0.62,
    "reason": "Relevant indirectly, but pricing power is not the main query focus."
  }
]
```

### Selected Primary SOP

```json
{
  "primary_sop": "Conditional Diligence Decision",
  "why": "The user asks for a decision: proceed with diligence or not. Other SOPs become supporting analytical frames."
}
```

### Selected secondary SOPs

```json
[
  {
    "sop": "Revenue Quality Evaluation",
    "role": "assess whether 85% growth is high-quality or volume-driven"
  },
  {
    "sop": "Customer Concentration Interpretation",
    "role": "assess whether top-three concentration is validation or fragility"
  },
  {
    "sop": "Platform vs Feature Evaluation",
    "role": "assess whether ERP bundling risk undermines defensibility"
  }
]
```

### Selected Sub-SOPs

```json
[
  {
    "sub_sop": "Software vs Transaction Revenue Separation",
    "from": "Revenue Quality Evaluation"
  },
  {
    "sub_sop": "Gross Margin Durability Check",
    "from": "Revenue Quality Evaluation"
  },
  {
    "sub_sop": "Fragile Dependency Check",
    "from": "Customer Concentration Interpretation"
  },
  {
    "sub_sop": "Strategic Account Validation Check",
    "from": "Customer Concentration Interpretation"
  },
  {
    "sub_sop": "Workflow Ownership Evaluation",
    "from": "Platform vs Feature Evaluation"
  },
  {
    "sub_sop": "Incumbent Bundling Risk Check",
    "from": "Platform vs Feature Evaluation"
  },
  {
    "sub_sop": "Diligence Gate Definition",
    "from": "Conditional Diligence Decision"
  }
]
```

---

## 5. Compact SOP Bundle

```json
{
  "bundle_id": "company_z_fintech_diligence_bundle",
  "primary_sop": {
    "name": "Conditional Diligence Decision",
    "core_policy": "Do not force a yes/no decision under mixed signals. Separate confirmed positives from unresolved assumptions, define proceed/pass conditions, and design the next diligence around the highest-impact uncertainties."
  },
  "supporting_sops": [
    {
      "name": "Revenue Quality Evaluation",
      "core_policy": "Do not treat headline growth as high quality until revenue composition, gross margin durability, and volume-driven cost scaling are understood."
    },
    {
      "name": "Customer Concentration Interpretation",
      "core_policy": "Interpret concentration contextually: top customers may validate product-market fit or expose fragile dependency."
    },
    {
      "name": "Platform vs Feature Evaluation",
      "core_policy": "Assess whether the company owns a persistent workflow platform or only a copyable feature vulnerable to ERP bundling."
    }
  ],
  "required_evidence_checklist": [
    "subscription vs payment-volume revenue breakdown",
    "gross margin by revenue type",
    "take-rate trend",
    "compliance/fraud/support cost scaling",
    "top-three customer expansion and renewal history",
    "gross margin by top customers",
    "customization burden for top customers",
    "pipeline of similar customers",
    "workflow ownership depth",
    "stored operational data, approval rules, and integrations",
    "ERP feature overlap and bundling risk"
  ],
  "failure_modes": [
    "Do not infer high-quality growth from 85% YoY growth alone.",
    "Do not call payment-volume revenue software revenue.",
    "Do not treat customer concentration as validation without expansion and margin evidence.",
    "Do not call the product a platform without workflow ownership and persistent data.",
    "Do not proceed with vague diligence; define specific gates."
  ],
  "sufficiency_criteria": [
    "Can classify revenue quality as software-like, transaction-driven, or mixed.",
    "Can classify customer concentration as validation, risk, or unresolved.",
    "Can assess platform vs feature risk against ERP bundling.",
    "Can define proceed conditions and pass conditions.",
    "Can explicitly list missing evidence."
  ],
  "answer_structure": [
    "direct decision",
    "positive signals",
    "major risks",
    "evidence gaps",
    "proceed conditions",
    "pass conditions",
    "confidence"
  ]
}
```

---

## 6. Query Decomposition into Search Tasks

The Reasoning & Planning module turns the SOP bundle into evidence-seeking tasks.

```json
{
  "search_tasks": [
    {
      "task": "Assess revenue quality",
      "search_queries": [
        "Company Z revenue breakdown subscription payment volume take rate",
        "Company Z gross margin by revenue type",
        "Company Z compliance fraud support cost transaction volume"
      ],
      "target_evidence": [
        "subscription vs transaction revenue",
        "take-rate trend",
        "gross margin durability",
        "cost scaling"
      ]
    },
    {
      "task": "Assess customer concentration",
      "search_queries": [
        "Company Z top three customers revenue share renewal expansion",
        "Company Z top customer gross margin customization discounts",
        "Company Z similar customer pipeline repeatability"
      ],
      "target_evidence": [
        "top customer expansion",
        "renewal history",
        "margin by customer",
        "customization burden",
        "repeatability"
      ]
    },
    {
      "task": "Assess platform vs feature",
      "search_queries": [
        "Company Z AP automation workflow ownership approval rules audit trail reconciliation",
        "Company Z ERP integration workflow data switching cost",
        "Company Z ERP vendor bundling feature overlap"
      ],
      "target_evidence": [
        "workflow depth",
        "persistent data",
        "integrations",
        "ERP bundling risk"
      ]
    },
    {
      "task": "Define diligence gates",
      "search_queries": [
        "Company Z diligence risks investment memo pass conditions",
        "Company Z proceed conditions revenue quality customer concentration ERP competition"
      ],
      "target_evidence": [
        "must-prove conditions",
        "pass conditions",
        "known risks"
      ]
    }
  ]
}
```

---

## 7. Mock Evidence Found

Since this is a fabricated demo, below is mock retrieved evidence in the format your system should produce.

```json
{
  "retrieved_evidence": [
    {
      "chunk_id": "doc_01_chunk_07",
      "source_type": "investment_memo",
      "summary": "Company Z revenue grew 85% YoY. Management estimates 38% of revenue is subscription workflow software, while 62% is tied to payment volume and take-rate."
    },
    {
      "chunk_id": "doc_02_chunk_03",
      "source_type": "financial_analysis",
      "summary": "Gross margin on subscription revenue is estimated at 78%, while payment-volume revenue has 34% gross margin after compliance review, payment failure handling, and support costs."
    },
    {
      "chunk_id": "doc_03_chunk_05",
      "source_type": "customer_analysis",
      "summary": "Top three customers represent 55% of revenue. Two of the three expanded from payment execution into invoice approval and reconciliation workflows over the last year."
    },
    {
      "chunk_id": "doc_04_chunk_02",
      "source_type": "customer_interview",
      "summary": "One large customer described Company Z as deeply integrated into AP approval routing and vendor onboarding, but another customer described it mainly as a cheaper payment-routing layer."
    },
    {
      "chunk_id": "doc_05_chunk_09",
      "source_type": "competitive_analysis",
      "summary": "Two ERP vendors recently launched bundled AP approval and payment-routing features. Their current products lack Company Z's cross-border payment coverage but overlap with basic workflow automation."
    },
    {
      "chunk_id": "doc_06_chunk_04",
      "source_type": "operations_memo",
      "summary": "Implementation for large customers requires 6–10 weeks of customization, mostly around approval rules, ERP integration, and compliance review workflows."
    },
    {
      "chunk_id": "doc_07_chunk_01",
      "source_type": "sales_pipeline",
      "summary": "Pipeline includes several similar mid-market customers, but conversion rates outside the top three accounts remain unproven."
    }
  ]
}
```

---

## 8. Structured Evidence Reorganization

```json
{
  "query": "Should we proceed with diligence on Company Z?",
  "selected_sop_bundle": "company_z_fintech_diligence_bundle",

  "supported_claims": [
    {
      "claim": "Company Z has strong headline growth, but revenue quality is mixed.",
      "supporting_evidence": [
        "doc_01_chunk_07",
        "doc_02_chunk_03"
      ],
      "reasoning": "85% YoY growth is positive, but 62% of revenue is payment-volume driven and carries much lower gross margin than subscription software.",
      "related_sops": [
        "Revenue Quality Evaluation",
        "Software vs Transaction Revenue Separation",
        "Gross Margin Durability Check"
      ],
      "confidence": "high"
    },
    {
      "claim": "Customer concentration is unresolved: it may indicate strategic validation, but it also creates fragility.",
      "supporting_evidence": [
        "doc_03_chunk_05",
        "doc_04_chunk_02"
      ],
      "reasoning": "Two top customers expanded into broader workflows, which is positive, but top-three customers still represent 55% of revenue and customer interpretation is inconsistent.",
      "related_sops": [
        "Customer Concentration Interpretation",
        "Strategic Account Validation Check",
        "Fragile Dependency Check"
      ],
      "confidence": "medium"
    },
    {
      "claim": "Company Z has some workflow-platform characteristics, but ERP bundling risk is real.",
      "supporting_evidence": [
        "doc_04_chunk_02",
        "doc_05_chunk_09",
        "doc_06_chunk_04"
      ],
      "reasoning": "Approval routing, vendor onboarding, reconciliation, and integrations suggest workflow depth. However, ERP vendors already overlap with basic workflow automation.",
      "related_sops": [
        "Platform vs Feature Evaluation",
        "Workflow Ownership Evaluation",
        "Incumbent Bundling Risk Check"
      ],
      "confidence": "medium"
    }
  ],

  "contradicting_or_cautionary_claims": [
    {
      "claim": "Company Z may be less software-like than headline growth implies.",
      "evidence": [
        "doc_01_chunk_07",
        "doc_02_chunk_03"
      ],
      "reasoning": "Most revenue is payment-volume driven, and payment revenue has much lower gross margin.",
      "confidence": "high"
    },
    {
      "claim": "Workflow lock-in may not be consistent across customers.",
      "evidence": [
        "doc_04_chunk_02"
      ],
      "reasoning": "One customer sees deep workflow integration; another sees the product mainly as payment routing.",
      "confidence": "medium"
    },
    {
      "claim": "ERP bundling could compress Company Z's differentiation.",
      "evidence": [
        "doc_05_chunk_09"
      ],
      "reasoning": "ERP vendors overlap with basic AP workflow automation, though not yet cross-border coverage.",
      "confidence": "medium"
    }
  ],

  "evidence_gaps": [
    "No clear renewal history for top customers.",
    "No customer-level gross margin by top account.",
    "No discounting or pricing trend.",
    "No quantified churn or NRR.",
    "No win/loss data versus ERP vendors.",
    "No evidence yet that similar customers convert repeatably outside the top three."
  ],

  "sop_coverage": {
    "Revenue Quality Evaluation": "strong",
    "Customer Concentration Interpretation": "partial",
    "Platform vs Feature Evaluation": "partial",
    "Conditional Diligence Decision": "sufficient for preliminary decision"
  },

  "preliminary_decision": "Proceed with targeted diligence, not full conviction."
}
```

---

## 9. Final Answer

### Conclusion

I would **proceed with targeted diligence**, but I would not yet treat Company Z as a high-conviction opportunity. The company has attractive surface-level growth, but the evidence suggests that revenue quality, customer concentration, and ERP bundling risk remain unresolved. `[doc_01_chunk_07]` `[doc_02_chunk_03]` `[doc_05_chunk_09]`

### Why the opportunity is interesting

Company Z may be more than a simple payment-routing product. Some evidence suggests that it touches deeper AP workflows such as approval routing, vendor onboarding, reconciliation, ERP integration, and compliance review. `[doc_04_chunk_02]` `[doc_06_chunk_04]`

The top customers also provide some positive signal. Two of the three largest customers expanded beyond payment execution into invoice approval and reconciliation workflows, which suggests possible workflow adoption. `[doc_03_chunk_05]`

### Why the evidence is not yet enough

The main issue is revenue quality. Although revenue grew 85% YoY, 62% of revenue is tied to payment volume and take-rate, while only 38% appears to be subscription workflow software. Payment-volume revenue also has much lower gross margin after compliance, payment failure handling, and support costs. `[doc_01_chunk_07]` `[doc_02_chunk_03]`

Customer concentration is also unresolved. Top three customers represent 55% of revenue. This could be acceptable if they are strategic accounts expanding profitably and repeatably, but the current evidence does not yet show renewal history, customer-level gross margin, discounting, or repeatability outside the top accounts. `[doc_03_chunk_05]` `[doc_07_chunk_01]`

The third issue is ERP bundling. ERP vendors have launched overlapping AP approval and payment-routing features. Company Z still appears differentiated in cross-border payment coverage, but if the core workflow layer is easy to bundle, the platform thesis is weaker. `[doc_05_chunk_09]`

### Proceed conditions

I would proceed only if the next diligence round can validate:

1. Subscription workflow revenue is growing faster than payment-volume revenue. `[doc_01_chunk_07]`
2. Gross margin improves as payment volume scales, rather than being consumed by compliance and support costs. `[doc_02_chunk_03]`
3. Top customers renew, expand, and remain profitable without heavy customization or discounting. `[doc_03_chunk_05]` `[doc_06_chunk_04]`
4. Similar mid-market customers can be acquired repeatably outside the top three accounts. `[doc_07_chunk_01]`
5. Company Z owns workflow data, approval rules, audit trails, integrations, or cross-border capabilities that ERP vendors cannot easily bundle away. `[doc_04_chunk_02]` `[doc_05_chunk_09]` `[doc_06_chunk_04]`

### Pass conditions

I would pass if diligence shows:

1. Growth is mainly payment-volume driven with weak software revenue. `[doc_01_chunk_07]`
2. Top customers are low-margin, highly customized, or have strong bargaining power. `[doc_03_chunk_05]` `[doc_06_chunk_04]`
3. The product is used mostly as a payment-routing layer rather than a workflow system of record. `[doc_04_chunk_02]`
4. ERP vendors can replicate the core workflow features with acceptable quality. `[doc_05_chunk_09]`
5. Sales outside the top three accounts are not repeatable. `[doc_07_chunk_01]`

### Confidence

**Confidence: Medium-low.**

The evidence supports continued diligence, but not conviction. The strongest current concern is that Company Z may look like a workflow software company while monetizing mostly like a lower-margin payments business. The next diligence round should focus on revenue composition, customer-level margin, renewal behavior, repeatability, and ERP competitive overlap. `[doc_01_chunk_07]` `[doc_02_chunk_03]` `[doc_03_chunk_05]` `[doc_05_chunk_09]`

---
