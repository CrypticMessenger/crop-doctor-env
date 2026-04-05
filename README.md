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

An OpenEnv RL environment where AI agents learn to diagnose crop diseases in Indian farms — investigating symptoms, running soil and lab tests, and recommending treatments under real-world budget and time constraints.

**India loses ₹50,000 crore annually to crop diseases.** With 60% of the population dependent on agriculture, timely diagnosis saves livelihoods. CropDoctorEnv trains AI agents to think like agricultural scientists — forming hypotheses, gathering evidence, and making treatment decisions through systematic investigation.

---

## Table of Contents

- [Environment Overview](#environment-overview)
- [Design Philosophy](#design-philosophy)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Tasks & Difficulty Progression](#tasks--difficulty-progression)
- [Reward Design](#reward-design)
- [Anti-Exploit Measures](#anti-exploit-measures)
- [Procedural Generation](#procedural-generation)
- [Baseline Scores](#baseline-scores)
- [Setup & Installation](#setup--installation)
- [Running Locally](#running-locally)
- [Running Inference](#running-inference)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Future Work (Round 2)](#future-work-round-2)

---

## Environment Overview

Each episode, a farmer reports a crop problem. The agent must figure out **why** and recommend the correct treatment. The cause is hidden — it could be a disease, pest, nutrient deficiency, or a combination. The agent has access to **21 diagnostic tools** across 7 categories, each with real costs (₹) and time (days).

```
Episode Flow:
  Farmer reports problem
       │
       ▼
  Agent investigates using tools
  (inspect leaves, test soil, send lab samples, check weather...)
       │
       ▼
  Agent submits diagnosis
  (identify problem + recommend specific treatment)
       │
       ▼
  Environment scores the diagnosis (0.0 – 1.0)
```

This mirrors the real workflow of a **Krishi Vigyan Kendra (KVK) agricultural scientist** — the same systematic investigation process used across India's 731 farm science centers.

### Inspiration

This environment is architecturally inspired by [BioExp](https://cerebralvalley.ai/e/openenv-hackathon-sf/hackathon/gallery), the 2nd-place winner ($9,000) at the Cerebral Valley OpenEnv Hackathon SF. BioExp trains agents to investigate unknown cancer cells using lab tools. CropDoctorEnv applies the same "hidden truth + diagnostic tools + resource constraints" pattern to agriculture — a domain with massive real-world impact in India.

---

## Design Philosophy

### Why This Environment Enables RL Training

We designed CropDoctorEnv around six principles that make it effective for reinforcement learning:

| Principle | Implementation | Why It Matters |
|-----------|---------------|----------------|
| **Dense rewards** | Per-step reward for every tool use (+0.03 to +0.15) | Agent knows immediately which actions help |
| **Reward shaping** | Partial credit for each correct finding | Smooth gradient toward optimal behavior |
| **Informative observations** | Full state visible: symptoms, budget, tools, findings | LLM can reason about trade-offs |
| **Structured actions** | 21 named tools with clear descriptions | Agent explores efficiently |
| **Progressive difficulty** | Easy (1 problem) → Medium (2) → Hard (3 + red herrings) | Curriculum learning |
| **Procedural generation** | Random crop × disease × soil each episode | Agent can't memorize — must reason |

### Resource Constraints (Real-World Modeling)

Every action has a cost, modeling real agricultural investigation:

```
Budget:     ₹10,000 total
Field Days: 7 days maximum
Lab Slots:  3 maximum (can't send unlimited samples)
```

This forces the agent to be **efficient** — it can't run every test. It must prioritize based on observed symptoms, just like a real agricultural scientist with limited resources.

### Action Dependencies (Scientific Methodology)

Actions have prerequisite rules that enforce proper diagnostic methodology:

```
❌ Cannot submit_diagnosis before at least 1 inspection + 1 paid test
❌ Cannot recommend_fungicide before identifying fungal disease
❌ Cannot send_leaf_sample before inspect_leaves (must collect first)
❌ Cannot compare_symptoms before at least 2 inspections
✅ Can inspect anything anytime
✅ Can check weather/database anytime
```

Violating a dependency returns a **hard penalty (-1.0)** and blocks the action. This teaches agents the correct order of operations — observe before testing, test before diagnosing.

---

## Action Space

The agent sends a `CropAction` with a tool name and optional parameters:

```python
class CropAction(Action):
    tool: str          # e.g., "inspect_leaves", "test_soil_npk", "submit_diagnosis"
    parameters: str    # Optional details, required for submit_diagnosis
```

### 21 Tools Across 7 Categories

| Category | Tool | Cost (₹) | Time (days) | Description |
|----------|------|-----------|-------------|-------------|
| **Field** | `inspect_leaves` | 0 | 0.1 | Visual inspection of leaf symptoms |
| | `inspect_stem` | 0 | 0.1 | Check stems for lesions, bore holes |
| | `inspect_roots` | 100 | 0.2 | Dig and examine root system |
| | `inspect_fruit` | 0 | 0.1 | Check fruits/grains for damage |
| | `check_pest_presence` | 0 | 0.1 | Look for visible insects/larvae |
| | `count_affected_plants` | 0 | 0.2 | Estimate % of field affected |
| **Soil** | `test_soil_ph` | 200 | 0.5 | Measure soil pH |
| | `test_soil_npk` | 500 | 0.5 | Test nitrogen, phosphorus, potassium |
| | `test_soil_micronutrients` | 800 | 1.0 | Test Zn, Fe, Mn, B, S, Mg levels |
| | `test_soil_moisture` | 100 | 0.2 | Measure soil moisture % |
| | `test_soil_type` | 0 | 0.1 | Identify soil classification |
| **Water** | `test_water_quality` | 300 | 0.5 | Test irrigation water salinity/pH |
| | `check_irrigation_status` | 0 | 0.2 | Check irrigation method/adequacy |
| **Weather** | `check_weather_history` | 0 | 0.1 | Last 30 days temperature/rain/humidity |
| | `check_weather_forecast` | 0 | 0.1 | Next 7 days forecast |
| **Lab** | `send_leaf_sample` | 2,000 | 2.0 | Lab pathogen identification |
| | `send_soil_sample` | 3,000 | 2.0 | Detailed soil lab analysis |
| | `identify_pest_species` | 1,500 | 1.0 | Lab pest species identification |
| **Knowledge** | `consult_crop_database` | 0 | 0.1 | Known diseases/pests for this crop |
| | `check_regional_alerts` | 0 | 0.1 | Current pest/disease advisories |
| | `compare_symptoms` | 0 | 0.1 | Match symptoms to disease profiles |
| **Terminal** | `submit_diagnosis` | 0 | 0.1 | Submit final diagnosis + treatment |

---

## Observation Space

After each action, the agent receives a `CropObservation`:

```python
class CropObservation(Observation):
    crop_info: str                  # "Crop: rice | Stage: vegetative | Soil: alluvial"
    tool_result: str                # Detailed result of last tool used
    findings_so_far: str            # Running summary of all discoveries
    available_tools: List[str]      # All tools the agent can use
    budget_remaining: int           # Rupees left
    days_remaining: float           # Field days left
    lab_slots_remaining: int        # Lab slots left
    step_number: int                # Current step
    message: str                    # Feedback + warnings (e.g., "⚠️ 3 steps remaining!")
```

The observation is designed to be **LLM-friendly** — all information is presented as clear text that any language model can parse and reason about.

---

## Tasks & Difficulty Progression

### Task 1: Easy — Single Problem, Clear Symptoms

- **Scenario**: One disease OR one pest, moderate severity
- **Complications**: None. No red herrings.
- **Example**: Rice with blast disease — diamond-shaped lesions clearly visible on leaves
- **Steps to solve**: ~5-8 (inspect → test → diagnose)
- **Max steps**: 15

### Task 2: Medium — Ambiguous Symptoms + Secondary Issue

- **Scenario**: One disease + one nutrient deficiency
- **Complications**: Symptoms overlap (yellowing could be rust OR nitrogen deficiency). One red herring (e.g., recent rain suggests waterlogging, but it's not the cause).
- **Example**: Wheat with rust disease + nitrogen deficiency — agent must run BOTH leaf inspection AND soil NPK test to differentiate
- **Steps to solve**: ~8-12
- **Max steps**: 20

### Task 3: Hard — Multiple Interacting Problems + Misleading Signals

- **Scenario**: One pest (severe) + one deficiency (moderate) + one disease (mild, hidden)
- **Complications**: The severe pest damage masks the mild disease. The deficiency makes the plant susceptible to both. Two red herrings present. Agent must discover ALL THREE problems.
- **Example**: Cotton with bollworm (obvious) + potassium deficiency (makes plant weak) + early wilt (hidden under pest damage)
- **Steps to solve**: ~12-18
- **Max steps**: 25

---

## Reward Design

### Per-Step Rewards (Dense Shaping Signal)

Every tool use returns an immediate reward:

| Signal | Reward | Trigger |
|--------|--------|---------|
| Info gain | +0.12 | Tool revealed information about an actual problem |
| Partial info | +0.06 | Tool found something useful (DETECTED/LOW) |
| Good ordering | +0.08 | Field inspection in first 3 steps |
| Novelty | +0.03 | Using a tool for the first time |
| Repetition | -0.05 | Using the same tool again |
| Wasted lab test | -0.08 | Expensive test found nothing relevant |
| Lazy investigation | -0.04 | Using knowledge tools before any inspection |
| Rule violation | -1.00 | Violating action dependency |

### Terminal Reward (Composite Score 0.0–1.0)

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Problem identification | 20-30% | Did agent name the correct disease/pest/deficiency? |
| Treatment quality | 20-30% | Did agent recommend the correct specific treatment? |
| Evidence chain | 15% | Did agent use enough diverse tools? |
| Budget efficiency | 10% | Spent enough to investigate but not wastefully? |
| Completeness bonus | 10% | Identified ALL problems (not just some)? |
| Spam penalty | -4% each | Mentioning diseases NOT supported by evidence |
| Violation penalty | -10% each | Rule violations during episode |

The weights shift by difficulty — easy rewards identification more heavily, hard requires more evidence.

---

## Anti-Exploit Measures

We tested 5 exploit strategies and patched all of them:

| Exploit | Strategy | Result |
|---------|----------|--------|
| Skip investigation | Submit diagnosis immediately | **Blocked** (-1.0): requires inspection + paid test |
| Keyword spam | Dump every disease name in diagnosis | **Penalized**: -4% per false mention |
| Free-tool only | Use only ₹0 tools, never spend budget | **Blocked** (-1.0): paid test required for submission |
| Tool repetition | Spam same free tool for step rewards | **Decays**: -0.05 per repeat, goes negative by step 4 |
| Minimal effort | 1 inspection + 1 test + keywords | **Low score** (0.16): insufficient evidence chain |

---

## Procedural Generation

Every episode is unique. The generator randomly selects:

- **Crop**: rice, wheat, cotton, tomato, potato, mustard (6 crops)
- **Growth stage**: seedling, vegetative, flowering, fruiting (crop-specific)
- **Soil type**: alluvial, black cotton, red laterite, sandy
- **Problems**: randomly selected from crop-specific disease/pest/deficiency pools
- **Red herrings**: water issues, unusual weather (medium/hard only)
- **Severity**: mild, moderate, severe

This produces **thousands of unique episodes** — the agent cannot memorize solutions. It must reason from symptoms each time.

### Data Coverage

- **17 diseases** with full symptom descriptions per plant part, pathogen identification, favorable conditions, and specific treatments
- **9 pests** with visual identification, damage patterns, and control measures
- **10 nutrient deficiencies** with symptom descriptions, soil test indicators, and fertilizer recommendations

All data is based on real agricultural science — disease symptoms, treatment protocols, and soil indicators match published agricultural extension literature.

---

## Baseline Scores

Tested with Qwen 2.5 72B Instruct (via HuggingFace Inference API):

| Task | Perfect Agent (scripted) | LLM Baseline (Qwen 72B) | Headroom |
|------|-------------------------|--------------------------|----------|
| Easy | 0.950 | 0.850 | 10% |
| Medium | 0.850 | 0.500 | 41% |
| Hard | 0.690 | 0.266 | 61% |

**Key observations:**
- The LLM correctly identifies obvious problems (aphid on easy, downy mildew on medium) but misses secondary issues (boron deficiency on medium, 2 of 3 problems on hard)
- The LLM sometimes violates action dependencies (tries `compare_symptoms` before `inspect_stem`) but learns from the penalty and corrects on the next step
- Significant headroom exists for RL training, especially on medium and hard tasks

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Docker (for containerized testing)

### Install

```bash
git clone https://github.com/CrypticMessenger/crop-doctor-env
cd crop-doctor-env
pip install openenv-core
pip install -r server/requirements.txt
```

---

## Running Locally

### Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Reset (start easy task)
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Step (use a tool)
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"tool": "inspect_leaves", "parameters": ""}'
```

### Docker

```bash
docker build -t crop-doctor-env .
docker run -p 8000:8000 crop-doctor-env
```

---

## Running Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
python inference.py
```

Expected output:
```
[START] task=easy env=crop_doctor_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=inspect_leaves reward=0.11 done=false error=null
[STEP] step=2 action=check_pest_presence reward=0.11 done=false error=null
...
[END] success=true steps=12 score=0.850 rewards=0.11,0.11,...
```

---

## Deployment

```bash
openenv push --repo-id celex4/crop-doctor-env
```

---

## Project Structure

```
crop_doctor_env/
├── models.py                    # CropAction, CropObservation, CropState (Pydantic)
├── client.py                    # CropDoctorEnv(EnvClient) — WebSocket client
├── openenv.yaml                 # Environment manifest
├── inference.py                 # Baseline LLM agent with [START]/[STEP]/[END] logs
├── Dockerfile
├── data/
│   ├── crops.json               # 6 crops with seasons, growth stages, disease/pest pools
│   ├── diseases.json            # 17 diseases — symptoms, pathogens, treatments
│   ├── pests.json               # 9 pests — symptoms, identification, control
│   └── deficiencies.json        # 10 nutrient deficiencies — symptoms, soil indicators
└── server/
    ├── app.py                   # FastAPI server (one-liner)
    ├── environment.py           # Core logic: reset(), step(), state()
    ├── generator.py             # Procedural episode generation
    ├── graders.py               # 7-term reward function with anti-exploit measures
    ├── tools.py                 # 21 tool implementations with costs/dependencies
    └── requirements.txt
```

---

## Future Work (Round 2)

If we advance to the finale, here are the techniques we plan to implement — informed by the latest research in agentic RL:

### 1. Code-Augmented LLM-as-Judge Verification (from AWM, ICML 2026)

The Agent World Model paper ([Wang et al., 2026](https://arxiv.org/html/2602.10090v2)) demonstrates that hybrid verification — combining code-based database state diffs with LLM trajectory reasoning — produces more robust reward signals than either approach alone. We plan to migrate CropDoctorEnv's state to SQLite and implement code-augmented verification: the grader inspects DB state before/after agent actions for deterministic checks, while an LLM judge evaluates reasoning quality from the trajectory. AWM showed this reduces both false positives and false negatives in reward assignment, which is critical for stable RL training.

### 2. GiGPO-Compatible Step-Level Credit Assignment (from GiGPO, NTU 2026)

Standard GRPO assigns one advantage per episode, making it hard to learn which specific tool choice was good or bad. GiGPO ([Feng et al., 2026](https://arxiv.org/html/2505.10978v1)) introduces "anchor state grouping" — when multiple trajectories encounter the same environment state, it compares what different actions did from that state, achieving +12% over GRPO on ALFWorld and +9% on WebShop with <0.002% overhead. CropDoctorEnv naturally produces anchor states (e.g., multiple episodes where the agent sees the same leaf symptoms), making it directly compatible with GiGPO's step-level credit assignment. We plan to validate this by training with GiGPO and reporting per-step advantage distributions.

### 3. History-Aware Training (from AWM)

AWM found that training with full interaction history but deploying with truncated history creates a distribution mismatch that hurts performance. Their solution: apply the same sliding-window truncation (w=3 turns) during training. We plan to implement this for CropDoctorEnv training, ensuring agents learn to make decisions with limited context — matching real-world deployment where agricultural agents may not have full investigation history available.

### 4. Adversarial Curriculum (inspired by BioExp)

Instead of fixed easy/medium/hard tasks, use an adversarial designer that generates scenarios targeting the agent's tracked weaknesses. If the agent keeps failing on fungal diseases, generate more fungal disease episodes. This is the technique that pushed BioExp to 2nd place ($9K) at the SF OpenEnv Hackathon.

### 5. Partial Symptom Masking

In the current version, all symptoms for all problems are visible simultaneously. In reality, a severe pest infestation masks a mild underlying disease. We plan to implement severity-based masking — the agent must clear the obvious problem first before deeper symptoms become visible, requiring multi-phase investigation.

### 6. GRPO/GiGPO Training Demonstration

Run actual training (using TRL + verl) on Qwen 1.5B and demonstrate score improvement over episodes. We aim to show learning curves comparable to BioExp (0.17 → 0.55 avg reward), proving the environment's reward signal is trainable. We will compare GRPO vs GiGPO to quantify the benefit of step-level credit assignment in our domain.

### 7. Real Map Integration

Integrate real Indian agricultural data via OpenStreetMap — village locations, soil type maps, regional crop patterns, and seasonal weather. Episodes generated for specific districts (e.g., "Kharif season in Guntur, Andhra Pradesh") making the environment directly applicable to real agricultural extension work.

---

## References

- **OpenEnv Framework**: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) — The environment spec this project implements
- **BioExp** (2nd place, OpenEnv Hackathon SF): Procedurally generated biological investigation environment with 40+ tools — architectural inspiration for CropDoctorEnv
- **AWM** (Wang et al., ICML 2026): [Agent World Model](https://arxiv.org/html/2602.10090v2) — Scalable environment synthesis with SQL-backed state and code-augmented verification
- **GiGPO** (Feng et al., 2026): [Group-in-Group Policy Optimization](https://arxiv.org/html/2505.10978v1) — Fine-grained step-level credit assignment for LLM agent training
- **ASTRA** (2026): [Automated Synthesis of Trajectories and Reinforcement Arenas](https://arxiv.org/html/2601.21558) — Verifiable reward RL for tool-use agents
- **TRL**: [Hugging Face TRL](https://huggingface.co/docs/trl) — GRPO implementation for LLM training

---

## License

MIT

## Team

Built for the Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, April 2026.
