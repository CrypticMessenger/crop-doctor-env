"""Tool implementations for CropDoctorEnv. Each tool takes the hidden episode
state and returns (result_text, cost_rupees, time_days)."""

import json, random
from typing import Tuple

Result = Tuple[str, int, float]  # (text, cost, days)


def _symptom_for(problems: list, part: str, diseases: dict, pests: dict, deficiencies: dict) -> str:
    """Aggregate symptoms across all active problems for a plant part."""
    lines = []
    for p in problems:
        name = p["name"]
        sev = p["severity"]
        if p["type"] == "disease" and name in diseases:
            s = diseases[name]["symptoms"].get(part, "")
        elif p["type"] == "pest" and name in pests:
            s = pests[name]["symptoms"].get(part, "")
        elif p["type"] == "deficiency" and name in deficiencies:
            s = deficiencies[name]["symptoms"].get(part, deficiencies[name]["symptoms"].get("growth", ""))
        else:
            s = ""
        if s:
            # Severity modulates description
            if sev == "mild":
                s = f"Mild signs: {s}"
            elif sev == "severe":
                s = f"Severe: {s}"
            lines.append(s)
    if not lines:
        return "Appears normal, no visible abnormalities."
    return " | ".join(lines)


def _add_noise(value: float, pct: float = 0.1) -> float:
    return round(value * (1 + random.uniform(-pct, pct)), 2)


# ── Field Observation Tools ──────────────────────────────────────────

def inspect_leaves(episode) -> Result:
    text = _symptom_for(episode["problems"], "leaves",
                        episode["diseases_db"], episode["pests_db"], episode["deficiencies_db"])
    # Add red herring symptoms if present
    for rh in episode.get("red_herrings", []):
        if rh == "unusual_weather":
            text += " | Note: some leaf tip browning possibly from recent heat wave."
    return text, 0, 0.1


def inspect_stem(episode) -> Result:
    return _symptom_for(episode["problems"], "stem",
                        episode["diseases_db"], episode["pests_db"], episode["deficiencies_db"]), 0, 0.1


def inspect_roots(episode) -> Result:
    return _symptom_for(episode["problems"], "roots",
                        episode["diseases_db"], episode["pests_db"], episode["deficiencies_db"]), 100, 0.2


def inspect_fruit(episode) -> Result:
    return _symptom_for(episode["problems"], "fruit",
                        episode["diseases_db"], episode["pests_db"], episode["deficiencies_db"]), 0, 0.1


def check_pest_presence(episode) -> Result:
    lines = []
    for p in episode["problems"]:
        if p["type"] == "pest":
            pest_data = episode["pests_db"].get(p["name"], {})
            vis = pest_data.get("symptoms", {}).get("pest_visible", "")
            if vis:
                lines.append(vis)
    if not lines:
        return "No significant pest activity observed.", 0, 0.1
    return " | ".join(lines), 0, 0.1


def count_affected_plants(episode) -> Result:
    total_pct = 0
    for p in episode["problems"]:
        base = {"mild": 15, "moderate": 40, "severe": 70}
        total_pct += base.get(p["severity"], 30)
    total_pct = min(95, total_pct)
    return f"Approximately {_add_noise(total_pct, 0.15):.0f}% of plants in the field show symptoms.", 0, 0.2


# ── Soil Testing Tools ───────────────────────────────────────────────

def test_soil_ph(episode) -> Result:
    base_ph = {"alluvial": 7.2, "black_cotton": 7.8, "red_laterite": 5.5, "sandy": 6.5}
    ph = _add_noise(base_ph.get(episode["soil_type"], 6.8), 0.05)
    return f"Soil pH: {ph:.1f} ({episode['soil_type']} soil)", 200, 0.5


def test_soil_npk(episode) -> Result:
    # Base levels, modified by deficiencies
    n, p, k = 280, 18, 150
    for prob in episode["problems"]:
        if prob["type"] == "deficiency":
            if prob["name"] == "nitrogen": n = random.randint(120, 200)
            elif prob["name"] == "phosphorus": p = random.randint(4, 8)
            elif prob["name"] == "potassium": k = random.randint(60, 100)
    n, p, k = _add_noise(n), _add_noise(p), _add_noise(k)
    status_n = "LOW" if n < 250 else "ADEQUATE"
    status_p = "LOW" if p < 10 else "ADEQUATE"
    status_k = "LOW" if k < 120 else "ADEQUATE"
    return (f"Soil NPK Analysis:\n"
            f"  Nitrogen (N): {n:.0f} kg/ha — {status_n}\n"
            f"  Phosphorus (P): {p:.0f} kg/ha — {status_p}\n"
            f"  Potassium (K): {k:.0f} kg/ha — {status_k}"), 500, 0.5


def test_soil_micronutrients(episode) -> Result:
    vals = {"zinc": 1.2, "iron": 6.0, "manganese": 3.5, "boron": 0.8,
            "sulphur": 15, "magnesium": 2.5}
    for prob in episode["problems"]:
        if prob["type"] == "deficiency" and prob["name"] in vals:
            # Make it low
            vals[prob["name"]] *= random.uniform(0.2, 0.4)
    lines = []
    for name, val in vals.items():
        v = _add_noise(val, 0.12)
        thresholds = {"zinc": 0.6, "iron": 4.5, "manganese": 2.0, "boron": 0.5, "sulphur": 10, "magnesium": 1.5}
        status = "LOW" if v < thresholds.get(name, 1.0) else "ADEQUATE"
        lines.append(f"  {name.capitalize()}: {v:.2f} ppm — {status}")
    return "Soil Micronutrient Analysis:\n" + "\n".join(lines), 800, 1.0


def test_soil_moisture(episode) -> Result:
    base = 35
    for rh in episode.get("red_herrings", []):
        if rh == "waterlogging":
            base = 85
        elif rh == "drought_stress":
            base = 12
    for p in episode["problems"]:
        if p["type"] == "deficiency" and p["name"] in ("drought_stress",):
            base = 10
    moisture = _add_noise(base, 0.15)
    status = "WATERLOGGED" if moisture > 70 else "DRY" if moisture < 20 else "ADEQUATE"
    return f"Soil moisture: {moisture:.0f}% — {status}", 100, 0.2


def test_soil_type(episode) -> Result:
    descriptions = {
        "alluvial": "Alluvial soil — fertile, well-drained, neutral to slightly alkaline",
        "black_cotton": "Black cotton (vertisol) — heavy clay, high water retention, cracks when dry",
        "red_laterite": "Red laterite — acidic, iron-rich, well-drained, low fertility",
        "sandy": "Sandy soil — light, well-drained, low water and nutrient retention",
    }
    return descriptions.get(episode["soil_type"], f"Soil type: {episode['soil_type']}"), 0, 0.1


# ── Water Testing Tools ──────────────────────────────────────────────

def test_water_quality(episode) -> Result:
    salinity = _add_noise(0.3, 0.3)
    ph = _add_noise(7.0, 0.08)
    for rh in episode.get("red_herrings", []):
        if rh == "salinity":
            salinity = _add_noise(2.5, 0.2)
    status = "SALINE" if salinity > 2.0 else "NORMAL"
    return f"Irrigation water: pH {ph:.1f}, EC {salinity:.2f} dS/m — {status}", 300, 0.5


def check_irrigation_status(episode) -> Result:
    methods = ["flood irrigation", "drip irrigation", "rainfed", "sprinkler"]
    method = random.choice(methods)
    adequacy = random.choice(["adequate", "insufficient", "excessive"])
    return f"Irrigation: {method}, frequency appears {adequacy}.", 0, 0.2


# ── Weather Tools ────────────────────────────────────────────────────

def check_weather_history(episode) -> Result:
    temp_avg = random.randint(22, 35)
    rain_mm = random.randint(0, 150)
    humidity = random.randint(50, 95)
    # Make weather match favorable conditions for the actual disease
    for p in episode["problems"]:
        if p["type"] == "disease":
            d = episode["diseases_db"].get(p["name"], {})
            cond = d.get("favorable_conditions", "")
            if "high humidity" in cond.lower():
                humidity = random.randint(85, 98)
            if "cool" in cond.lower():
                temp_avg = random.randint(15, 22)
    return (f"Last 30 days weather:\n"
            f"  Avg temperature: {temp_avg}°C\n"
            f"  Total rainfall: {rain_mm}mm\n"
            f"  Avg humidity: {humidity}%"), 0, 0.1


def check_weather_forecast(episode) -> Result:
    temp = random.randint(22, 38)
    rain_chance = random.randint(10, 80)
    return f"Next 7 days: Avg {temp}°C, {rain_chance}% chance of rain.", 0, 0.1


# ── Lab Testing Tools (expensive, slow) ─────────────────────────────

def send_leaf_sample(episode) -> Result:
    results = []
    for p in episode["problems"]:
        if p["type"] == "disease":
            d = episode["diseases_db"].get(p["name"], {})
            pathogen = d.get("pathogen", "Unknown pathogen")
            dtype = d.get("type", "unknown")
            results.append(f"DETECTED: {pathogen} ({dtype} pathogen) — confirms {p['name']}")
    if not results:
        results.append("No significant pathogens detected in leaf tissue.")
    return "Lab Report (Leaf Sample):\n  " + "\n  ".join(results), 2000, 2.0


def send_soil_sample(episode) -> Result:
    lines = ["Lab Report (Detailed Soil Analysis):"]
    lines.append(f"  Soil type: {episode['soil_type']}")
    lines.append(f"  Organic carbon: {_add_noise(0.5, 0.3):.2f}%")
    for p in episode["problems"]:
        if p["type"] == "deficiency":
            db = episode["deficiencies_db"].get(p["name"], {})
            indicator = db.get("soil_indicator", {})
            lines.append(f"  ALERT: {p['name'].capitalize()} — {json.dumps(indicator)}")
    return "\n".join(lines), 3000, 2.0


def identify_pest_species(episode) -> Result:
    results = []
    for p in episode["problems"]:
        if p["type"] == "pest":
            results.append(f"IDENTIFIED: {p['name'].replace('_', ' ').title()} — severity: {p['severity']}")
    if not results:
        results.append("No significant pest species identified.")
    return "Lab Report (Pest Identification):\n  " + "\n  ".join(results), 1500, 1.0


# ── Knowledge Tools (free) ───────────────────────────────────────────

def consult_crop_database(episode) -> Result:
    crop = episode["crop"]
    crop_data = episode["crops_db"].get(crop, {})
    diseases = crop_data.get("diseases", [])
    pests = crop_data.get("pests", [])
    deficiencies = crop_data.get("common_deficiencies", [])
    return (f"Database for {crop} ({crop_data.get('season', '')} crop):\n"
            f"  Common diseases: {', '.join(diseases)}\n"
            f"  Common pests: {', '.join(pests)}\n"
            f"  Common deficiencies: {', '.join(deficiencies)}"), 0, 0.1


def check_regional_alerts(episode) -> Result:
    # Randomly include the actual problem in alerts (helpful clue)
    alerts = []
    for p in episode["problems"]:
        if random.random() > 0.4:  # 60% chance the real problem is in alerts
            alerts.append(f"Advisory: {p['name'].replace('_', ' ')} reported in nearby districts")
    if not alerts:
        alerts.append("No active pest/disease advisories for this region.")
    return "Regional Alerts:\n  " + "\n  ".join(alerts), 0, 0.1


def compare_symptoms(episode) -> Result:
    """Requires at least 2 inspections to have been done (checked by environment)."""
    crop = episode["crop"]
    crop_data = episode["crops_db"].get(crop, {})
    matches = []
    for p in episode["problems"]:
        if p["type"] == "disease":
            matches.append(f"Symptoms consistent with: {p['name']} ({p['severity']})")
        elif p["type"] == "pest":
            matches.append(f"Damage pattern consistent with: {p['name']} ({p['severity']})")
        elif p["type"] == "deficiency":
            matches.append(f"Nutrient symptoms consistent with: {p['name']} deficiency ({p['severity']})")
    # Add a possible false match for medium/hard
    if len(episode["problems"]) > 1 and random.random() > 0.5:
        decoy = random.choice(crop_data.get("diseases", ["unknown"]))
        matches.append(f"Partial match (low confidence): {decoy}")
    return "Symptom Comparison Results:\n  " + "\n  ".join(matches), 0, 0.1


# ── Tool Registry ────────────────────────────────────────────────────

TOOL_REGISTRY = {
    # Field observation
    "inspect_leaves": {"fn": inspect_leaves, "category": "field", "needs_prior": []},
    "inspect_stem": {"fn": inspect_stem, "category": "field", "needs_prior": []},
    "inspect_roots": {"fn": inspect_roots, "category": "field", "needs_prior": []},
    "inspect_fruit": {"fn": inspect_fruit, "category": "field", "needs_prior": []},
    "check_pest_presence": {"fn": check_pest_presence, "category": "field", "needs_prior": []},
    "count_affected_plants": {"fn": count_affected_plants, "category": "field", "needs_prior": []},
    # Soil
    "test_soil_ph": {"fn": test_soil_ph, "category": "soil", "needs_prior": []},
    "test_soil_npk": {"fn": test_soil_npk, "category": "soil", "needs_prior": []},
    "test_soil_micronutrients": {"fn": test_soil_micronutrients, "category": "soil", "needs_prior": []},
    "test_soil_moisture": {"fn": test_soil_moisture, "category": "soil", "needs_prior": []},
    "test_soil_type": {"fn": test_soil_type, "category": "field", "needs_prior": []},
    # Water
    "test_water_quality": {"fn": test_water_quality, "category": "water", "needs_prior": []},
    "check_irrigation_status": {"fn": check_irrigation_status, "category": "field", "needs_prior": []},
    # Weather
    "check_weather_history": {"fn": check_weather_history, "category": "weather", "needs_prior": []},
    "check_weather_forecast": {"fn": check_weather_forecast, "category": "weather", "needs_prior": []},
    # Lab (expensive)
    "send_leaf_sample": {"fn": send_leaf_sample, "category": "lab", "needs_prior": ["inspect_leaves"]},
    "send_soil_sample": {"fn": send_soil_sample, "category": "lab", "needs_prior": ["test_soil_npk"]},
    "identify_pest_species": {"fn": identify_pest_species, "category": "lab", "needs_prior": ["check_pest_presence"]},
    # Knowledge
    "consult_crop_database": {"fn": consult_crop_database, "category": "knowledge", "needs_prior": []},
    "check_regional_alerts": {"fn": check_regional_alerts, "category": "knowledge", "needs_prior": []},
    "compare_symptoms": {"fn": compare_symptoms, "category": "knowledge", "needs_prior": ["inspect_leaves", "inspect_stem"]},
    # Diagnosis (terminal)
    "submit_diagnosis": {"fn": None, "category": "terminal", "needs_prior": ["_any_inspection", "_any_test"]},
}

# Descriptions shown to the agent
TOOL_DESCRIPTIONS = {
    "inspect_leaves": "Visually inspect leaves for symptoms (free, 0.1 day)",
    "inspect_stem": "Visually inspect stems for symptoms (free, 0.1 day)",
    "inspect_roots": "Dig and inspect roots (₹100, 0.2 day)",
    "inspect_fruit": "Inspect fruits/grains for damage (free, 0.1 day)",
    "check_pest_presence": "Look for visible pests and insects (free, 0.1 day)",
    "count_affected_plants": "Estimate % of field affected (free, 0.2 day)",
    "test_soil_ph": "Test soil pH level (₹200, 0.5 day)",
    "test_soil_npk": "Test soil N, P, K levels (₹500, 0.5 day)",
    "test_soil_micronutrients": "Test Zn, Fe, Mn, B, S, Mg levels (₹800, 1 day)",
    "test_soil_moisture": "Measure soil moisture (₹100, 0.2 day)",
    "test_soil_type": "Identify soil type (free, 0.1 day)",
    "test_water_quality": "Test irrigation water quality (₹300, 0.5 day)",
    "check_irrigation_status": "Check irrigation method and adequacy (free, 0.2 day)",
    "check_weather_history": "Get last 30 days weather data (free, 0.1 day)",
    "check_weather_forecast": "Get 7-day weather forecast (free, 0.1 day)",
    "send_leaf_sample": "Send leaf to lab for pathogen ID (₹2000, 2 days) — requires inspect_leaves first",
    "send_soil_sample": "Send soil to lab for detailed analysis (₹3000, 2 days) — requires test_soil_npk first",
    "identify_pest_species": "Lab identification of pest species (₹1500, 1 day) — requires check_pest_presence first",
    "consult_crop_database": "Look up known diseases/pests for this crop (free, 0.1 day)",
    "check_regional_alerts": "Check current pest/disease advisories (free, 0.1 day)",
    "compare_symptoms": "Match observed symptoms to disease profiles (free, 0.1 day) — requires 2+ inspections",
    "submit_diagnosis": "Submit final diagnosis and treatment plan — requires at least 1 inspection + 1 test",
}
