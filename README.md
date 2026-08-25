# V5 — Data Mixture & Curriculum

**Target:** ~1T-total / 32B-active Mixture-of-Experts, trained from scratch on **9.5T tokens**. **Goal capabilities:** agentic coding, controllable reasoning depth, native Indic fluency.

This document fixes the share of the token budget for every capability lane, the Indic split across four provenance tiers, the stage-by-stage curriculum, the protected floors the data selector may not cross, the anneal reserve, and the proxy experiments that must validate all of it before full scale.

**One rule governs everything below: no lane is sized beyond the tokens that actually exist for it.**

## Conventions

- **Every number in the main plan is reachable from data that exists today.** Nothing here depends on data we have not yet acquired or generated.
- Repetition ceiling: **≤ 4 epochs on any corpus.** [Muennighoff et al., NeurIPS 2023](https://arxiv.org/abs/2305.16264) find up to 4 epochs of repeated data yields negligible loss change versus fresh data; value decays toward zero well beyond that.
- Supply figures come from the course dataset inventory. Where a public source disagrees, the discrepancy is noted in §12.
- "Synthetic" in the Indic lane means **Sangraha-synthetic, a published AI4Bharat dataset**. Using it is not us generating data. The same applies to OpenMathReasoning, OpenThoughts2 and OpenR1-Math — all pre-existing public releases.
- Any data we would have to *generate ourselves* is excluded from the plan and confined to **Appendix A**, which is a proposal, not a commitment.

## 1\. Premises

**Architecture.** ~1T total parameters, 32B active, fine-grained experts plus a shared expert. The closest public analogue is [Kimi K2](https://arxiv.org/abs/2507.20534) — 1.04T total / 32B active, 384 experts with 8 routed + 1 shared, trained on 15.5T tokens. Sparsity ratio ~3.2%.

**Token budget: 9.5T.** MoE scaling is anchored to *active* parameters, not total, so 9.5T ÷ 32B ≈ **297 tokens per active parameter**. This sits far above Chinchilla-optimal (20×) and below Kimi K2 (484×) and DeepSeek-V3 (~400×) — deliberately, because we are data-constrained, not compute-constrained.

**Supply is the hard cap.**

| Lane | Unique supply |
| --- | --- |
| General web & STEM | 4.84T |
| Code | 1.10T |
| Indic | 276B |
| Long-context | 100B |
| Reasoning & math | 85.1B |
| Agentic & tool-use | 0.627B |
| **Total unique** | **~6.4T** |

9.5T over 6.4T unique = **1.48 average epochs**, comfortably inside the safe-repetition regime. But the average hides the problem: two lanes are catastrophically short, and the 4-epoch ceiling turns that shortage into a hard budget limit.

**Maximum feasible lane size at 4 epochs:**

| Lane | Supply | 4-epoch ceiling | As % of 9.5T |
| --- | --- | --- | --- |
| General web & STEM | 4.84T | 19.3T | 100%+ |
| Code | 1.10T | 4.41T | 46.4% |
| Indic | 276B | 1.10T | 11.6% |
| Long-context | 100B | 400B | 4.2% |
| Reasoning & math | 85.1B | 340B | **3.6%** |
| Agentic & tool-use | 0.627B | 2.5B | **0.03%** |

This table is the whole argument. Reasoning cannot exceed 3.6% and agentic cannot exceed 0.03% of pretraining without either acquiring more real data or generating it. The mixture below respects those ceilings instead of wishing them away.

## 2\. The mixture

| Lane | Share | Tokens | Supply | Epochs | Fed by | Benchmarks it must win |
| --- | --- | --- | --- | --- | --- | --- |
| **General web & STEM** | 58.9% | 5,597B | 4.84T | 1.16 | DCLM-Baseline 2.6T, FineWeb-Edu 1.3T, D2 627B, D1 164B, D4-STEM 49B, proof-pile-2 55B, peS2o 42B | MMLU-Pro, GPQA-Diamond |
| **Code** | 25.3% | 2,400B | 1.10T | 2.18 | The Stack v2 900B, D3 Code 199B, CommitPackFT 4B | SWE-bench Verified, LiveCodeBench, Terminal-Bench |
| **Indic** *(protected)* | 8.0% | 760B | 276B | 2.75 | Sangraha (verified / unverified / synthetic), IndicCorpV2, Samanantar, BPCC | IndicGenBench, MILU, IN22, FLORES-200 |
| **Long-context** | 4.2% | 400B | 100B | 4.00 | Repo-packed code 60B, book-length corpora 40B | RULER, long-doc QA |
| **Reasoning & math** | 3.6% | 340B | 85.1B | 4.00 | AON 78B, OpenThoughts2 3B, OpenMathReasoning 2B, OpenR1-Math 1.6B, NuminaMath 0.5B | AIME, MATH-500, GPQA-Diamond |
| **Agentic & tool-use** *(protected)* | 0.03% | 2.5B | 0.627B | 4.00 | ToolBench, Glaive v2, ToolACE, xLAM/APIGen, Nexus, SWE-smith, Hermes, OpenHands, SWE-Gym | BFCL v3, τ-bench, SWE-bench Verified |
| **Total** | **100.0%** | **9,500B** |  |  |  |  |

### Why these numbers

**General at 58.9%** is higher than Llama 3's published ~50% general share. That is a *consequence*, not a preference: with reasoning capped at 3.6% and agentic at 0.03%, the freed budget has to go somewhere, and General and Code are the only lanes with the supply to absorb it.

**Code at 25.3%** exceeds Llama 3's 17% because agentic coding is the flagship capability and code is the one specialist lane with abundant supply (1.1T). At 2.18 epochs it is well inside the ceiling and could go higher if proxy runs justify it.

**Reasoning at 3.6% is the ceiling, not a choice.** 85.1B × 4 epochs = 340B. This is the tightest constraint in the plan after agentic. Most reasoning capability is instilled *after* pretraining anyway — via SFT on reasoning traces and RL with verifiable rewards — so a thin pretraining lane is less damaging than it looks. It remains a real limitation; Appendix A addresses it.

**Agentic at 0.03% is an honest failure to meet the requirement.** See §7.

## 3\. Indic lane — tier split

Lane total **760B**, split across four provenance tiers. Sangraha's own composition is published as verified 64B / unverified 24B / synthetic 162B, totalling 251B across 22 languages ([IndicLLMSuite](https://arxiv.org/abs/2403.06350)).

| Tier | Source | Share of lane | Tokens | Supply | Epochs |
| --- | --- | --- | --- | --- | --- |
| **Synthetic** | Sangraha-synthetic | 40.5% | 308B | 162B | 1.90 |
| **Verified** | Sangraha-verified | 33.4% | 254B | 64B | 3.97 |
| **Unverified** | Sangraha-unverified + IndicCorpV2 | 23.4% | 178B | 44.9B | 3.96 |
| **Translated** | Samanantar + BPCC | 2.6% | 20B | 5B | 4.00 |
| **Total** |  | **100.0%** | **760B** | 276B | 2.75 |

**Synthetic is capped at 40%**, meaning 60% of the Indic lane is human-authored. The justification is the model-collapse literature, which splits cleanly on one variable:

- [Shumailov et al.,](https://www.nature.com/articles/s41586-024-07566-y) *[Nature](https://www.nature.com/articles/s41586-024-07566-y)* [631 (2024)](https://www.nature.com/articles/s41586-024-07566-y) — when synthetic data **replaces** real data across generations, models collapse.
- [Gerstgrasser et al. (arXiv:2404.01413)](https://arxiv.org/abs/2404.01413) — when synthetic data **accumulates alongside** real data, test error has a finite upper bound independent of iteration count. Collapse does not occur.
- [Dohmatob et al.,](https://arxiv.org/abs/2410.04840) *[Strong Model Collapse](https://arxiv.org/abs/2410.04840)* [(ICLR 2025)](https://arxiv.org/abs/2410.04840) — the cautionary counterpoint: in some regimes even a small synthetic fraction degrades performance, particularly when synthetic data is unverified and uncurated.

We are in the accumulation regime by construction — real Indic data is never removed, and synthetic is a fixed minority — but Dohmatob's result is why the cap is 40% rather than 60%, and why the anneal is verified-only.

**Translated is deliberately small (2.6%).** Samanantar and BPCC are parallel corpora, valuable for cross-lingual transfer but carrying translationese risk for native generation. A 5B supply also means anything larger would breach the epoch ceiling.

### Tier composition by stage

Synthetic front-loaded, verified rising monotonically, anneal almost entirely verified.

| Stage | Indic budget | Synthetic | Verified | Unverified | Translated |
| --- | --- | --- | --- | --- | --- |
| S0 Warmup | 8B | 3B (37.5%) | 2B (25.0%) | 2B (25.0%) | 1B (12.5%) |
| S1 Foundation | 448B | 240B (53.6%) | 86B (19.2%) | 110B (24.6%) | 12B (2.7%) |
| S2 Capability ramp | 241B | 65B (26.9%) | 109B (45.2%) | 61B (25.4%) | 6B (2.5%) |
| S3 Long-context | 3B | — | 2B (73.3%) | 0.7B (26.7%) | — |
| S4 Anneal | 60B | — | 55B (91.7%) | 4B (6.7%) | 1B (1.7%) |
| **Total** | **760B** | **308B** | **254B** | **178B** | **20B** |

Verified share by stage: **25% → 19% → 45% → 73% → 92%.** The model meets Indic in bulk early and in quality late.

## 4\. Curriculum

Warmup-Stable-Decay schedule. WSD decouples the token budget from the LR schedule, so the stable-phase checkpoint can be branched to test multiple anneal mixes without re-running the trunk.

| Stage | Tokens | % of run | Seq len | LR phase |
| --- | --- | --- | --- | --- |
| **S0** Warmup | 100B | 1.1% | 4K | 0 → peak |
| **S1** General foundation | 5,600B | 58.9% | 4K | stable |
| **S2** Capability ramp | 3,245B | 34.2% | 4K | stable |
| **S3a** Context extension | 40B | 0.4% | 4K→128K | stable |
| **S3b** Context extension | 15B | 0.2% | 128K→256K | early decay |
| **S4** Anneal / cooldown | 500B | 5.3% | 32K | decay → ~0 |
| **Total** | **9,500B** | **100.0%** |  |  |

### Per-stage mixtures

**S0 — Warmup (100B).** Indic enters at its full floor from token one, so Indic scripts are present while embeddings are still forming. This is the structural fix for the V4 failure where a late Hindi-share jump into frozen embeddings produced a ~150× gradient-norm spike.

| Lane | % | Tokens |
| --- | --- | --- |
| General | 84.0% | 84B |
| Code | 8.0% | 8B |
| Indic | 8.0% | 8B |
| Agentic | 0.03% | 0.03B |
| **Total** | **100.0%** | **100B** |

**S1 — General foundation (5,600B).** Broad language and world knowledge.

| Lane | % | Tokens |
| --- | --- | --- |
| General | 69.3% | 3,880B |
| Code | 17.9% | 1,002B |
| Indic | 8.0% | 448B |
| Long-context | 3.0% | 168B |
| Reasoning | 1.8% | 100B |
| Agentic | 0.03% | 1.7B |
| **Total** | **100.0%** | **5,600B** |

**S2 — Capability ramp (3,245B).** General falls, code and reasoning rise.

| Lane | % | Tokens |
| --- | --- | --- |
| General | 45.5% | 1,478B |
| Code | 38.6% | 1,252B |
| Indic | 7.4% | 241B |
| Reasoning | 4.3% | 140B |
| Long-context | 4.1% | 134B |
| Agentic | 0.02% | 0.6B |
| **Total** | **100.0%** | **3,245B** |

**S3a + S3b — Context extension (55B).** Long-context dominant; see §5.

| Lane | % | Tokens |
| --- | --- | --- |
| Long-context | 70.0% | 38.5B |
| Code | 15.0% | 8.3B |
| General | 10.0% | 5.5B |
| Indic | 5.0% | 2.7B |
| Agentic | 0.03% | 0.01B |
| **Total** | **100.0%** | **55B** |

**S4 — Anneal (500B).** Reserve data only; LR decays to ~0.

| Lane | % | Tokens |
| --- | --- | --- |
| General (top-quality) | 30.0% | 150B |
| Code | 26.0% | 130B |
| Reasoning | 20.0% | 100B |
| Indic (verified-dominant) | 12.0% | 60B |
| Long-context | 12.0% | 60B |
| Agentic | 0.03% | 0.15B |
| **Total** | **100.0%** | **500B** |

Across the run: **General 69% → 46% → 30%**, **Code 18% → 39% → 26%**, **Reasoning 1.8% → 4.3% → 20%**.

### Transition blending

No mixture change is applied as a step function. Each boundary is blended linearly across a warmup band:

| Transition | Blend band |
| --- | --- |
| S0 → S1 | 50B |
| S1 → S2 | 100B |
| S2 → S3a | 30B (plus RoPE rescaling warmup) |
| S3b → S4 | 20B |

Indic never moves more than its floor across any boundary, so the V4 gradient spike cannot recur by construction.

## 5\. Context length: 128K → 256K

**Answer to "are we capped at 128K?" — no.** The target is **256K**, reached in a mid-training extension stage, then exercised in post-training. Long context is not a from-scratch pretraining property; every frontier model acquires it late.

| Model | Context | How it was reached |
| --- | --- | --- |
| Llama 3.1 | 128K | 6-stage extension, ~800B tokens, late pretraining |
| DeepSeek-V3 | 128K | Two-phase YaRN: 4K→32K, then 32K→128K |
| Qwen3 | 256K native | Native long pretrain + YaRN extrapolation |
| Kimi K2 | 128K → 256K | Extended after initial release |

The capability is cheap to acquire. [Fu et al. (ICML 2024)](https://arxiv.org/abs/2402.10171) show **500M–5B tokens** suffice for retrieval anywhere in a 128K window, provided you use **per-source length upsampling** — drawing longer sequences within each domain while holding the domain mixture constant — rather than globally upweighting book-like domains, which degrades short-context ability. [ProLong](https://arxiv.org/abs/2410.02660) reaches 128K in **40B tokens**.

**Our schedule:**

| Phase | Seq len | Method | Budget |
| --- | --- | --- | --- |
| S0–S2 | 4K | RoPE base 10K | 8.9T |
| S3a | 4K → 32K → 128K | RoPE base scaling (ABF), then YaRN | 40B |
| S3b | 128K → 256K | YaRN scale-up | 15B |
| S4 | 32K train, 256K retained | — | within anneal |
| Post-train SFT | up to 256K | short SFT + limited long SFT | ≤1B |

**We validate effective length, not claimed length.** [RULER](https://arxiv.org/abs/2404.06654) finds almost all models fall below threshold before their advertised context; one analysis puts Llama 3.1 70B's effective length at ~64K against a 128K claim. Gate: run RULER at 32K / 64K / 128K / 256K after S3a, S3b and SFT, including multi-needle and variable-tracking tasks — not single-needle, which flatters everything. **We do not ship a 256K label unless RULER passes at 256K.**

### A note on double-counting

Roughly 100B of the long-context lane is genuinely unique long-document supply. The rest of the long sequences are produced by **re-packing code and books already counted in the Code and General lanes** — dependency-ordered repo packing (as in DeepSeek-Coder) and length-aware document packing. Those tokens are counted once, in their home lane. The long-context lane is a *packing and ordering* decision layered on existing supply, not 400B of new text.

## 6\. Protected floors

A global data selector optimizes an aggregate objective and will starve small lanes, because low-resource groups contribute little to the average. Two lanes are therefore carved out **before** the selector runs, and the selector optimizes only the remainder.

| Lane | Floor | Enforcement |
| --- | --- | --- |
| **Indic** | 8.0% of every micro-batch | Reserved slot, outside selector control, present in every stage including warmup and anneal |
| **Agentic** | 0.03% of every micro-batch | Same mechanism; never dropped, buffer replayed if it underflows |

The floor mechanism is not novel — continual-pretraining work finds that **replaying as little as 1% of prior data substantially mitigates forgetting**. A small, *consistent* presence beats a large, intermittent one.

Both floors ramp in across the S0 warmup band rather than switching on instantly.

## 7\. The agentic problem, stated plainly

**The requirement was an always-on agentic lane. The plan delivers 0.03%. That is a two-orders-of-magnitude shortfall and it should be the first thing a reviewer pushes on.**

The arithmetic: 627M real agentic tokens × 4 epochs = 2.5B. Against 9.5T that is 0.026%. There is no way to reach 3–5% from real supply.

**What makes this survivable:**

1. **Tool-use is primarily a post-training capability.** Kimi K2 — the model this design mirrors — does its large-scale agentic data work in SFT and RL, not pretraining. The pretraining lane's job is format familiarity, not capability.
2. **The floor still does real work.** At 0.03% always-on, the model sees function-call syntax, JSON tool schemas and short ReAct traces continuously, so the format is never foreign when post-training begins.
3. **Long-horizon trajectories belong later anyway.** SWE-Gym trajectories average ~305K tokens each; these are RL-environment material, not pretraining shards.

**What makes it worse:** the public executable-environment supply is genuinely small. SWE-smith is the largest single source at ~50K instances; SWE-Gym has 2,438. Even aggressive acquisition of every public agentic dataset moves the lane from 0.03% to perhaps 0.1%.

**Conclusion:** agentic capability in V5 will be won in post-training or not at all. Appendix A sets out what generating agentic data would require, if the team decides the gap must be closed in pretraining.

## 8\. Anneal reserve

**500B tokens (5.3% of the run) held back in a locked manifest.** Reserve shards carry a `reserve=true` flag; both the S0–S3 dataloader and the selector filter them out, so they are physically un-sampleable until S4 unlocks them. Without this mechanism the best data is simply consumed early and there is nothing special left to anneal on.

| Source | Reserved |
| --- | --- |
| proof-pile-2 | 55B |
| peS2o | 42B |
| Curated reasoning (OpenThoughts2, OpenMathReasoning, NuminaMath, OpenR1-Math) | ~7B |
| Sangraha-verified, top shards | 55B |
| Highest-quality agentic traces | 0.15B |
| DCLM / FineWeb-Edu top quality percentile | remainder to 500B |

Precedent: Llama 3 reports that annealing on small amounts of high-quality code and math data measurably improves benchmarks, and that tokens at the end of training carry disproportionate weight.

## 9\. Difficulty and reasoning-length bands

### Difficulty ladders, with a real example at each level

**Reasoning & math**

1. GSM8K grade-school word problem
2. MATH competition problem
3. AIME item (integer answer, multi-step)
4. AoPS olympiad problem
5. Proof-requiring item (proof-pile-2, Lean-formalizable)

**Code**

1. HumanEval / MBPP single function
2. LiveCodeBench competitive problem
3. Real bug-fix patch (CommitPackFT)
4. SWE-bench Verified multi-file patch — e.g. `django__django-11099`, *"ModelBackend.authenticate() shouldn't make a database query when username is None"*, graded by hidden FAIL*TO*PASS and PASS*TO*PASS tests
5. Repo-scale refactor / Terminal-Bench shell task

**Agentic & tool-use**

1. Single function call (BFCL simple)
2. Parallel / multiple calls (BFCL v2)
3. Multi-turn with state (BFCL v3, τ-bench)
4. Long-horizon with failure recovery (τ²-bench, OpenHands rollout ~9K tokens)
5. Full SWE-Gym trajectory (~305K tokens)

**Indic**

1. Short native sentence (IndicCorpV2)
2. Paragraph-level native text (Sangraha)
3. Cross-lingual transfer / MT (IN22: 1,024 sentences × 22 languages; FLORES-200)
4. Indic reasoning (MILU: multi-domain MCQ across 11 languages)
5. Indic generation and instruction-following (IndicGenBench: CrossSum-IN, Flores-IN, XQuAD-IN, XorQA-IN)

**Long-context**

1. Single-needle retrieval
2. Multi-needle retrieval
3. Variable tracking / aggregation (RULER task categories)
4. Multi-document QA
5. Repo-scale code comprehension at 128K–256K

**General & STEM**

1. MMLU
2. MMLU-Pro
3. GPQA-Diamond — graduate-level, "Google-proof"; domain PhDs reach ~65%, skilled non-experts ~34% with unrestricted web access
4. Humanity's Last Exam

### Reasoning-length bands

Appropriate reasoning length differs by task type. Short chain-of-thought outperforms long CoT under tight budgets; long CoT wins only as difficulty and budget rise. Overthinking measurably flips correct answers to incorrect on easy items.

| Lane | Low (<256 tok) | Medium (256–1K) | High (1K–4K) | Ultra (4K–16K) |
| --- | --- | --- | --- | --- |
| Reasoning & math | 10% | 30% | 40% | 20% |
| Code | 25% | 40% | 25% | 10% |
| Agentic | 20% | 35% | 30% | 15% |
| Indic | 45% | 40% | 12% | 3% |
| Long-context | 30% | 40% | 25% | 5% |
| General & STEM | 40% | 40% | 15% | 5% |

Every row sums to 100%. Indic and General skew short because most items are knowledge and retrieval rather than multi-step derivation. The low/medium/high/ultra dial itself is learned in post-training, via length-controlled RL — the inference-time setting selects among behaviours the model was trained to produce; it does not create them.

## 10\. Loss masking

| Data type | Loss applied to | Context only, masked |
| --- | --- | --- |
| Pretraining (S0–S4) | All tokens | — |
| Instruction SFT | Assistant response | System, user prompt |
| **Agentic trajectories** | **Assistant reasoning and tool calls** | **User turns, tool observations, environment returns** |
| Distilled reasoning traces | Reasoning + final answer | Problem statement |
| Long-context | Target span | Retrieved context |

The agentic rule is the one that matters. Applying loss to tool observations teaches the model to **invent tool results instead of calling tools** — the single most damaging masking error in agentic training.

## 11\. Proxy validation

**The mixture above is a hypothesis. None of it is trusted at 1T until it survives proxy runs.**

**Arms — 1B active, 30B tokens each:**

| Arm | Variation | Primary metric | Decision rule |
| --- | --- | --- | --- |
| A0 | Natural distribution baseline | Aggregate downstream | Reference |
| A1 | The §2 mixture | Aggregate downstream | Promote only if ≥ A0 |
| A2 | A1 with protected floors disabled | Indic + tool-format probes | If A2 ≪ A1 on protected lanes, floors are justified |
| A3 | Indic synthetic share at 30 / 40 / 50 / 60% | MILU, IndicGenBench | Select highest share before quality inflects |
| A4 | Code at 20% vs 30% | LiveCodeBench proxy | Locate the code knee |
| A5 | Reasoning at 2% vs 3.6%, General absorbing the difference | MATH-500 proxy | Confirm the 3.6% ceiling is not costing more than it appears |

**Then:** re-run the two surviving mixtures at **3B active, 60B tokens** to test rank stability.

**Cost:** 1B × 30B ≈ 1.8e20 FLOP ≈ ~125 GPU-hours per arm; 3B × 60B ≈ 1.08e21 ≈ ~750 GPU-hours per arm. Six 1B arms plus two 3B arms ≈ **2,250 GPU-hours, under 0.1% of the full run.**

**Additional gates:**

- If tool-format retention on a held-out BFCL-simple probe drops more than 2 points with the agentic floor at 0.03%, raise the floor and accept more epochs.
- If RULER effective length at 128K after S3a is below 96K, double the S3a budget before attempting S3b.
- If the Indic verified tier shows validation-loss divergence at 4 epochs, cap it at 3.5 and shift the difference to synthetic, staying under the 40% cap.

**Caveat, stated up front:** small-proxy rankings do not reliably transfer. Mixture rankings can invert between scales, and some capabilities are invisible at 1B. Proxy results are a prior to be re-estimated at 3B, not a proof. The WSD checkpoint stays branchable precisely so the anneal mix can be re-decided late with better information.

## 12\. Risks

| Risk | Trigger | Mitigation |
| --- | --- | --- |
| Reasoning lane too thin at 3.6% | Weak AIME / MATH proxy results | Reasoning is mostly post-training; if proxies fail, acquire more real reasoning data (Appendix A) |
| Agentic lane effectively absent | BFCL / τ-bench failure after SFT | Accept post-training as the delivery path; Appendix A if unacceptable |
| Indic verified & unverified at the 4-epoch ceiling | Validation-loss uptick | Cap at 3.5 epochs, shift to synthetic under the 40% cap |
| Synthetic Indic quality drift | Factuality or fluency degradation | 60% real retained; verified-only anneal; A3 arm tests the cap directly |
| Gradient-norm spike at transitions | Loss or grad-norm jump at a boundary | Blend bands (§4); Indic present from token one; skip-batch on spike |
| MoE router collapse under distribution shift | Expert utilization skew at a stage boundary | Load-balancing loss + router z-loss; monitor per-expert utilization at every transition |
| 256K claimed but not effective | RULER multi-needle failure | Do not ship the label; extend S3 budget |
| Benchmark contamination | Leakage into the anneal reserve | Exclude known benchmark train sets from reserve; time-cutoff filtering for code |

## Appendix A — Optional data-generation roadmap

*A proposal, not part of the plan. Nothing in §1–§12 depends on any of it.*

This appendix exists because two lanes are supply-starved, and a reviewer will reasonably ask what closing the gap would take. If the team later decides to generate data, this is the shape of the work and its cost.

### A.1 What the field actually does

| Approach | Generator | Verifier | Published scale |
| --- | --- | --- | --- |
| Answer-verified CoT distillation | Strong reasoning teacher | Rule-based answer match | OpenMathReasoning: 5.5M SFT samples from ~306K seed problems |
| Rejection-sampling fine-tuning (STaR / RFT) | Model samples many CoTs | Keep only answer-correct traces | Standard across math pipelines |
| Rephrasing / style augmentation | Any capable model | None required | Kimi K2 reports 10× rephrasing + 1 epoch beating 10 raw epochs on SimpleQA; WRAP reports ~3× pretraining speedup |
| Verified function-call synthesis | LLM over an API pool | Three-stage: format → execution → semantic | APIGen: 60K verified entries over 3,673 executable APIs |
| Executable SWE task synthesis | Auto-built environments per repo | Unit-test pass | SWE-smith: ~50K instances / 128 repos, 5,016 expert trajectories |
| Agentic trajectory synthesis | Simulated users + real tool execution | Rubric-based LLM judge | Kimi K2 pipeline (volume unpublished) |

The unifying lesson: **synthetic data is safe in proportion to how well it is verified.** Execution checks and answer matching are strong verifiers; unverified generation is exactly the regime where the collapse results apply.

### A.2 What it would cost us

**Reasoning — lifting the lane from 3.6% to ~8.7% (826B tokens).** Roughly 650B additional verified tokens needed. Verified yield on hard math is well under 100%, so assume a 3–5× generation multiplier → 2–3T raw generated tokens. With a 32B-active teacher at 2N FLOP per token, that is \*\*1.3–2.1e23 FLOP of teacher inference\*\* — a material fraction of the pretraining run itself. It must be budgeted as a line item, not assumed free.

**Agentic — lifting the lane from 0.03% to ~3%.** ~285B tokens needed. At 50–100K tokens per verified trajectory that is 3–6M verified trajectories; at a 3–5× rollout multiplier, 10–30M rollouts. The entire public executable-environment ecosystem is on the order of tens of thousands of task instances. **The gap is two to three orders of magnitude.** This is not a compute problem; it is an environment-supply problem, and it is why the field does agentic work in RL rather than pretraining.

### A.3 If we do any of it, in what order

1. **Rephrasing existing corpora** — cheapest, best-evidenced, no new environments required. Applies to Indic and reasoning immediately.
2. **Answer-verified reasoning distillation** — strong verifier, well-trodden path, directly addresses the tightest ceiling after agentic.
3. **Verified function-call synthesis** — moderate cost, closes part of the agentic format gap.
4. **Executable SWE environments** — highest cost, highest value, and the only path to genuine long-horizon agentic data.

Each would need its own proxy arm before entering the mixture. A generated corpus is a hypothesis exactly like a mixture proportion is.