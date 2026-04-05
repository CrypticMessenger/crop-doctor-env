"""Procedural episode generation for CropDoctorEnv."""

import json, random, uuid
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def _load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())

CROPS_DB = _load_json("crops.json")
DISEASES_DB = _load_json("diseases.json")
PESTS_DB = _load_json("pests.json")
DEFICIENCIES_DB = _load_json("deficiencies.json")

SOIL_TYPES = ["alluvial", "black_cotton", "red_laterite", "sandy"]
WATER_ISSUES = ["drought_stress", "waterlogging", "salinity"]


def generate_episode(difficulty: str = "easy", seed: int | None = None) -> dict:
    """Generate a random episode. Returns full hidden state dict."""
    if seed is not None:
        random.seed(seed)

    crop = random.choice(list(CROPS_DB.keys()))
    crop_info = CROPS_DB[crop]
    soil = random.choice(SOIL_TYPES)
    stage = random.choice(crop_info["growth_stages"])

    if difficulty == "easy":
        ptype = random.choice(["disease", "pest"])
        if ptype == "disease":
            name = random.choice(crop_info["diseases"])
            problems = [{"type": "disease", "name": name, "severity": "moderate"}]
        else:
            name = random.choice(crop_info["pests"])
            problems = [{"type": "pest", "name": name, "severity": "moderate"}]
        red_herrings = []
        budget, days, lab_slots, max_steps = 10000, 7.0, 3, 15

    elif difficulty == "medium":
        disease = random.choice(crop_info["diseases"])
        deficiency = random.choice(crop_info.get("common_deficiencies", list(DEFICIENCIES_DB.keys())))
        problems = [
            {"type": "disease", "name": disease, "severity": "moderate"},
            {"type": "deficiency", "name": deficiency, "severity": "mild"},
        ]
        red_herrings = [random.choice(WATER_ISSUES)]
        budget, days, lab_slots, max_steps = 10000, 7.0, 3, 20

    else:  # hard
        pest = random.choice(crop_info["pests"])
        deficiency = random.choice(crop_info.get("common_deficiencies", list(DEFICIENCIES_DB.keys())))
        disease = random.choice(crop_info["diseases"])
        problems = [
            {"type": "pest", "name": pest, "severity": "severe"},
            {"type": "deficiency", "name": deficiency, "severity": "moderate"},
            {"type": "disease", "name": disease, "severity": "mild"},
        ]
        red_herrings = [random.choice(WATER_ISSUES), "unusual_weather"]
        budget, days, lab_slots, max_steps = 10000, 7.0, 3, 25

    # Compute ground truth treatments
    treatments = []
    for p in problems:
        if p["type"] == "disease":
            t = DISEASES_DB.get(p["name"], {}).get("treatment", {})
        elif p["type"] == "pest":
            t = PESTS_DB.get(p["name"], {}).get("treatment", {})
        elif p["type"] == "deficiency":
            t = DEFICIENCIES_DB.get(p["name"], {}).get("treatment", {})
        else:
            t = {}
        treatments.append({"problem": p["name"], "type": p["type"], "treatment": t})

    return {
        "episode_id": str(uuid.uuid4()),
        "difficulty": difficulty,
        "crop": crop,
        "growth_stage": stage,
        "soil_type": soil,
        "problems": problems,
        "red_herrings": red_herrings,
        "ground_truth_treatments": treatments,
        "budget": budget,
        "days": days,
        "lab_slots": lab_slots,
        "max_steps": max_steps,
        # Attach DBs for tool lookups
        "crops_db": CROPS_DB,
        "diseases_db": DISEASES_DB,
        "pests_db": PESTS_DB,
        "deficiencies_db": DEFICIENCIES_DB,
    }
