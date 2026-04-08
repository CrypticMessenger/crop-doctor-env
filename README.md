---
title: CropDoctorEnv
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
---

# 🌾 CropDoctorEnv: AI Agricultural Disease Diagnosis

### Can an LLM learn to be a crop doctor — from scratch?

We gave a language model a farmer's complaint, 21 diagnostic tools, ₹10,000, and 7 days. No agricultural training data. No few-shot examples. Just wilting leaves and a budget.

It had to learn what every agricultural scientist at India's 731 Krishi Vigyan Kendras already knows: **observe first, test second, diagnose last.** And it had to learn it from reward signal alone.

**India loses ₹50,000 crore annually to crop diseases.** 60% of the population depends on agriculture. A single misdiagnosis — treating a fungal infection with pesticide, or missing a hidden nutrient deficiency under obvious pest damage — can destroy an entire season's harvest. CropDoctorEnv trains AI agents to think like agricultural scientists: forming hypotheses, gathering evidence under real-world constraints, and making treatment decisions that could save livelihoods.

> Built with [OpenEnv v0.2](https://github.com/meta-pytorch/OpenEnv) | Deployed on [HF Spaces](https://huggingface.co/spaces/celex4/crop-doctor-env)

---

## The Story: From Guessing to Diagnosing

### Act 1: The Blind Farmer Visit

Episode 1. The agent receives its first case: *"Farmer reports problems with rice crop. Vegetative stage, alluvial soil."*

It has never seen a farm. It doesn't know what blast disease looks like, that nitrogen deficiency yellows leaves from the bottom up, or that stem borers leave bore holes you can only find by splitting the stem open. It tries `submit_diagnosis` immediately. **Blocked.** The environment enforces scientific methodology — you can't diagnose what you haven't investigated. Reward: **-1.0**.

### Act 2: Learning to Look

Episode 4. The agent discovers that `inspect_leaves` is free and fast. It sees *"diamond-shaped lesions with grey centers on leaf blades."* It doesn't know what this means yet, but it notices the reward went positive (+0.12). It tries `inspect_stem` next. More positive reward. It's learning: **look before you test.**

### Act 3: The Budget Trap

Episode 8. The agent has learned to inspect everything. But now it's on a medium-difficulty case — wheat with rust AND nitrogen deficiency. It sends a leaf sample to the lab (₹2,000). It sends a soil sample (₹3,000). It runs micronutrient tests (₹800). Budget: ₹4,200 remaining. It needs more tests but can't afford them.

The environment taught it something real agricultural scientists learn the hard way: **you can't test everything. You must prioritize based on what you've already seen.**

### Act 4: The Red Herring

Episode 12. Hard difficulty. Cotton with bollworm (severe), potassium deficiency (moderate), and grey mildew (mild, hidden). The agent sees obvious pest damage and immediately focuses on bollworm. It identifies the pest correctly. But the environment only gives partial credit — there are two more problems hiding under the pest damage.

A red herring appears: recent rainfall suggests waterlogging. The agent wastes a step testing water quality. The soil is fine. The real issue is potassium — the plant's weakened immune system let the fungus in.

**The best agents learn to ask: "What else could be wrong?"**

### Act 5: The Anti-Exploit Arms Race

We tested every shortcut an RL agent might find:

- **Keyword spam**: Dump every disease name in the diagnosis → **Penalized** (-4% per false mention)
- **Skip investigation**: Submit immediately → **Blocked** (requires inspection + paid test)
- **Free-tool farming**: Only use ₹0 tools → **Blocked** (paid test required for submission)
- **Tool repetition**: Spam `inspect_leaves` for step rewards → **Decays** to negative by step 4

The environment fights back. Every exploit we found, we patched. The agent must actually learn agriculture.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CROP DOCTOR EPISODE LOOP                        │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Procedural   │    │  Hidden       │    │  Agent               │  │
│  │  Generator    │───►│  Ground Truth │───►│  (LLM)               │  │
│  │  (crop×disease│    │  (diseases,   │    │  Sees: symptoms,     │  │
│  │   ×soil×pest) │    │   pests,      │    │  budget, tools       │  │
│  └──────────────┘    │   deficiency) │    │  Chooses: 1 of 21    │  │
│                      └──────┬───────┘    │  diagnostic tools     │  │
│                             │            └──────────┬───────────┘  │
│                             │                       │              │
│                             ▼                       ▼              │
│                      ┌──────────────┐    ┌──────────────────────┐  │
│                      │  Tool Engine  │◄──│  Action Dependencies │  │
│                      │  (21 tools,   │    │  (inspect before     │  │
│                      │   real costs, │    │   test, test before  │  │
│                      │   noisy data) │    │   diagnose)          │  │
│                      └──────┬───────┘    └──────────────────────┘  │
│                             │                                      │
│                             ▼                                      │
│                      ┌──────────────────────────────────────────┐  │
│                      │  7-Term Reward Function                  │  │
│                      │  ├─ Info gain (+0.12 per relevant find)  │  │
│                      │  ├─ Ordering bonus (inspect early)       │  │
│                      │  ├─ Novelty (+0.03 new tool)             │  │
│                      │  ├─ Efficiency (budget sweet spot)       │  │
│                      │  ├─ Completeness (found ALL problems?)   │  │
│                      │  ├─ Spam penalty (-4% per false claim)   │  │
│                      │  └─ Violation penalty (-10% each)        │  │
│                      └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### The Loop

1. **Procedural Generator** creates a unique episode: random crop × disease × soil × complications. Thousands of combinations — the agent can never memorize.
2. **Hidden Ground Truth** contains the actual problems (disease, pest, deficiency), their severity, and correct treatments. The agent cannot see this.
3. **Agent** receives a farmer's complaint and must investigate using 21 tools — each with real costs (₹) and time (days).
4. **Tool Engine** executes the chosen tool against the hidden state, returning noisy observations (soil pH ±10%, symptom descriptions that overlap between diseases).
5. **Action Dependencies** enforce scientific methodology — can't diagnose before investigating, can't send lab samples before field inspection.
6. **7-Term Reward** scores every action and the final diagnosis, with anti-exploit measures that penalize gaming.

### What Makes This Different

- **Real agricultural science** — 17 diseases, 9 pests, 10 deficiencies with accurate symptoms, pathogens, and treatments from Indian agricultural extension literature
- **Resource constraints model reality** — ₹10,000 budget, 7 field days, 3 lab slots. A real KVK scientist faces the same trade-offs.
- **Noisy observations** — Soil pH readings vary ±5%, plant counts vary ±15%. The agent must reason under uncertainty.
- **Red herrings** — Medium/hard tasks include misleading signals (recent rain → "waterlogging?" but it's actually a nutrient issue)
- **Anti-exploit by design** — Every shortcut we found, we patched. The agent must actually learn diagnostic reasoning.

---

## Action Space

21 tools across 7 categories. Each has a real cost and time:

| Category | Tool | Cost (₹) | Time (days) | Prerequisites |
|----------|------|-----------|-------------|---------------|
| **Field** | `inspect_leaves` | 0 | 0.1 | — |
| | `inspect_stem` | 0 | 0.1 | — |
| | `inspect_roots` | 100 | 0.2 | — |
| | `inspect_fruit` | 0 | 0.1 | — |
| | `check_pest_presence` | 0 | 0.1 | — |
| | `count_affected_plants` | 0 | 0.2 | — |
| **Soil** | `test_soil_ph` | 200 | 0.5 | — |
| | `test_soil_npk` | 500 | 0.5 | — |
| | `test_soil_micronutrients` | 800 | 1.0 | — |
| | `test_soil_moisture` | 100 | 0.2 | — |
| | `test_soil_type` | 0 | 0.1 | — |
| **Water** | `test_water_quality` | 300 | 0.5 | — |
| | `check_irrigation_status` | 0 | 0.2 | — |
| **Weather** | `check_weather_history` | 0 | 0.1 | — |
| | `check_weather_forecast` | 0 | 0.1 | — |
| **Lab** | `send_leaf_sample` | 2,000 | 2.0 | `inspect_leaves` |
| | `send_soil_sample` | 3,000 | 2.0 | `test_soil_npk` |
| | `identify_pest_species` | 1,500 | 1.0 | `check_pest_presence` |
| **Knowledge** | `consult_crop_database` | 0 | 0.1 | — |
| | `check_regional_alerts` | 0 | 0.1 | — |
| | `compare_symptoms` | 0 | 0.1 | 2+ inspections |
| **Terminal** | `submit_diagnosis` | — | — | 1 inspection + 1 paid test |

---

## Observation Space

```python
class CropObservation(Observation):
    crop_info: str              # "Crop: rice | Stage: vegetative | Soil: alluvial"
    tool_result: str            # Detailed result of last tool
    findings_so_far: str        # Running summary of all discoveries
    available_tools: List[str]  # All 21 tools
    budget_remaining: int       # ₹ left
    days_remaining: float       # Field days left
    lab_slots_remaining: int    # Lab slots left (max 3)
    step_number: int
    message: str                # Feedback + "⚠️ 3 steps remaining!"
```

Designed to be **LLM-friendly** — any language model can parse the observation and reason about what to do next. No hidden state, no encoded vectors. Just clear text.

---

## Tasks & Difficulty Progression

| Task | Problems | Red Herrings | Max Steps | What Agent Must Learn |
|------|----------|-------------|-----------|----------------------|
| **Easy** | 1 disease OR 1 pest | None | 15 | Basic investigation flow |
| **Medium** | 1 disease + 1 deficiency | 1 (water issue) | 20 | Differentiate overlapping symptoms |
| **Hard** | 1 pest + 1 deficiency + 1 disease | 2 (water + weather) | 25 | Find hidden problems under obvious ones |

---

## Reward Design

### Per-Step (Dense Shaping)

| Signal | Reward | Trigger |
|--------|--------|---------|
| Info gain | +0.12 | Tool revealed actual problem info |
| Partial info | +0.06 | Tool found something useful |
| Good ordering | +0.08 | Field inspection in first 3 steps |
| Novelty | +0.03 | First use of a tool |
| Repetition | -0.05 | Same tool again |
| Wasted lab test | -0.08 | Expensive test, nothing found |
| Wrong order | -0.04 | Knowledge tools before inspection |
| Rule violation | -1.00 | Dependency violation |

### Terminal (Composite 0.0–1.0)

| Component | Weight | Measures |
|-----------|--------|----------|
| Problem identification | 20-30% | Named the correct problems? |
| Treatment quality | 20-30% | Correct specific treatment? |
| Evidence chain | 15% | Enough diverse tools used? |
| Budget efficiency | 10% | Spent wisely (not too little, not too much)? |
| Completeness | 10% | Found ALL problems? |
| Spam penalty | -4% each | False disease mentions |
| Violation penalty | -10% each | Rule violations |

---

## Procedural Generation

Every episode is unique:

- **6 crops**: rice, wheat, cotton, tomato, potato, mustard
- **17 diseases**: blast, bacterial blight, rust, wilt, leaf curl, early/late blight...
- **9 pests**: stem borer, bollworm, aphid, whitefly, tuber moth...
- **10 deficiencies**: nitrogen, phosphorus, potassium, zinc, iron...
- **4 soil types**: alluvial, black cotton, red laterite, sandy
- **4 growth stages**: seedling, vegetative, flowering, fruiting

Thousands of unique combinations. The agent cannot memorize — it must reason from symptoms every time.

---

## Baseline Scores

Tested with Qwen 2.5 72B Instruct via HuggingFace Inference API:

| Task | Score | Steps | Budget Used | Key Behavior |
|------|-------|-------|-------------|-------------|
| **Easy** | 0.650 | 4 | ₹2,000 | Identified bacterial blight correctly, submitted quickly |
| **Medium** | 0.345 | 5 | ₹2,100 | Found sheath rot but missed nutrient deficiency entirely |
| **Hard** | 0.327 | 6 | ₹3,500 | Found leaf folder + bacterial blight, missed deficiency |

**The agent submits too early.** It diagnoses after 4-6 steps without running soil tests, so it consistently misses nutrient deficiencies. This is exactly the behavior RL training would fix — teaching the agent to investigate deeper before concluding.

---

## Quick Start

```python
from client import CropDoctorEnv
from models import CropAction

async with CropDoctorEnv(base_url="https://celex4-crop-doctor-env.hf.space") as env:
    result = await env.reset(task_id="easy")
    print(result.observation.crop_info)      # "Crop: rice | Stage: vegetative | Soil: alluvial"
    print(result.observation.tool_result)     # "A farmer has reported problems..."

    result = await env.step(CropAction(tool="inspect_leaves", parameters=""))
    print(result.observation.tool_result)     # "Diamond-shaped lesions with grey centers..."

    result = await env.step(CropAction(tool="test_soil_npk", parameters=""))
    result = await env.step(CropAction(tool="submit_diagnosis",
        parameters="Blast disease caused by Magnaporthe oryzae. Apply tricyclazole fungicide."))
    print(result.reward)                      # 0.72
```

---

## Running Inference

```bash
export OPENAI_API_KEY="your-key"       # or HF_TOKEN / API_KEY
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

Output:
```
[START] task=easy env=crop_doctor_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=inspect_leaves reward=0.11 done=false error=null
[STEP] step=2 action=check_pest_presence reward=0.11 done=false error=null
...
[END] success=true steps=12 score=0.850 rewards=0.11,0.11,...
```

---

## Deployment

Deployed as a Docker-based HF Space. To run locally:

```bash
docker build -t crop-doctor-env .
docker run -p 7860:7860 crop-doctor-env
```

The server will be available at `http://localhost:7860`.

---

## Project Structure

```
crop_doctor_env/
├── models.py              # CropAction, CropObservation, CropState (Pydantic)
├── client.py              # CropDoctorEnv(EnvClient) — async client
├── inference.py           # Baseline LLM agent with [START]/[STEP]/[END] logs
├── openenv.yaml           # OpenEnv manifest
├── Dockerfile
├── data/
│   ├── crops.json         # 6 crops: seasons, growth stages, disease/pest pools
│   ├── diseases.json      # 17 diseases: symptoms per plant part, pathogens, treatments
│   ├── pests.json         # 9 pests: visual ID, damage patterns, control measures
│   └── deficiencies.json  # 10 deficiencies: symptoms, soil indicators, fertilizers
└── server/
    ├── app.py             # FastAPI server
    ├── environment.py     # Core: reset(), step(), state()
    ├── generator.py       # Procedural episode generation
    ├── graders.py         # 7-term reward with anti-exploit measures
    └── tools.py           # 21 tool implementations with costs/dependencies
```

---

## Future Work (Round 2)

### Adversarial Curriculum
Instead of fixed easy/medium/hard, use an LLM designer that generates scenarios targeting the agent's tracked weaknesses.

### GiGPO Step-Level Credit Assignment
Standard GRPO assigns one advantage per episode. [GiGPO (Feng et al., 2026)](https://arxiv.org/html/2505.10978v1) compares actions from the same state across rollouts, achieving +12% over GRPO on ALFWorld. CropDoctorEnv naturally produces anchor states (same leaf symptoms → different tool choices), making it directly compatible.

### Code-Augmented Verification
Migrate state to SQLite and implement hybrid verification: code checks DB state diffs, LLM judge evaluates reasoning quality. Based on [AWM (Wang et al., ICML 2026)](https://arxiv.org/html/2602.10090v2).

### Partial Symptom Masking
Severe pest damage currently coexists with mild disease symptoms. In reality, the obvious problem masks the hidden one. Implement severity-based masking — clear the obvious problem first, then deeper symptoms become visible.

### Real Map Integration
Integrate real Indian agricultural data — village locations, soil maps, regional crop patterns, seasonal weather. Episodes generated for specific districts (e.g., "Kharif season in Guntur, Andhra Pradesh").

---

## References

- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv) — Environment spec
- [GiGPO](https://arxiv.org/html/2505.10978v1) — Step-level credit assignment for LLM agents
- [AWM](https://arxiv.org/html/2602.10090v2) — Code-augmented verification for RL environments
- [HF TRL](https://huggingface.co/docs/trl) — GRPO implementation

---

MIT License | Built for Meta PyTorch OpenEnv Hackathon × Scaler, April 2026
