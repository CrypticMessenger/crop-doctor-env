"""Multi-term reward computation for CropDoctorEnv.

v3 fixes:
- Anti keyword-spam: penalize mentioning diseases/pests NOT found in evidence
- Budget efficiency: spending ₹0 is NOT optimal — must spend SOME to investigate
- Evidence quality: tools used must actually relate to the problems found
"""

import json

DIFFICULTY_CONFIG = {
    "easy":   {"id_weight": 0.30, "treat_weight": 0.30, "evidence_min": 3, "min_budget_pct": 0.05},
    "medium": {"id_weight": 0.25, "treat_weight": 0.25, "evidence_min": 5, "min_budget_pct": 0.10},
    "hard":   {"id_weight": 0.20, "treat_weight": 0.20, "evidence_min": 8, "min_budget_pct": 0.15},
}

# All known problem names for spam detection
ALL_DISEASE_NAMES = {
    "blast", "bacterial_blight", "sheath_rot", "rust", "smut", "karnal_bunt",
    "wilt", "leaf_curl", "grey_mildew", "early_blight", "late_blight",
    "mosaic_virus", "black_scurf", "common_scab", "white_rust",
    "alternaria_blight", "downy_mildew",
}
ALL_PEST_NAMES = {
    "stem_borer", "leaf_folder", "aphid", "armyworm", "bollworm",
    "whitefly", "fruit_borer", "tuber_moth", "painted_bug",
}
ALL_DEFICIENCY_NAMES = {
    "nitrogen", "phosphorus", "potassium", "zinc", "iron",
    "manganese", "sulphur", "boron", "calcium", "magnesium",
}
ALL_PROBLEM_NAMES = ALL_DISEASE_NAMES | ALL_PEST_NAMES | ALL_DEFICIENCY_NAMES


def compute_step_reward(tool_name: str, result_text: str, episode: dict, tools_used: list) -> float:
    """Per-step shaped reward."""
    reward = 0.0
    problems = episode["problems"]
    problem_names = {p["name"] for p in problems}
    result_lower = result_text.lower()

    # Info gain: tool revealed something about an actual problem
    found_relevant = False
    for pname in problem_names:
        if pname.replace("_", " ") in result_lower or pname in result_lower:
            found_relevant = True
            break

    if found_relevant:
        reward += 0.12
    elif "DETECTED" in result_text or "IDENTIFIED" in result_text or "LOW" in result_text:
        reward += 0.06

    # Ordering bonus: field inspection in first 3 steps
    field_tools = {"inspect_leaves", "inspect_stem", "inspect_roots", "inspect_fruit", "check_pest_presence"}
    if tool_name in field_tools and len(tools_used) <= 3:
        reward += 0.08

    # Novelty
    if tools_used.count(tool_name) == 1:
        reward += 0.03

    # Penalty: repeating
    if tools_used.count(tool_name) > 1:
        reward -= 0.05

    # Penalty: expensive test found nothing
    if tool_name.startswith("send_") or tool_name == "identify_pest_species":
        if not found_relevant:
            reward -= 0.08

    # Penalty: knowledge tools before inspection
    knowledge_tools = {"consult_crop_database", "check_regional_alerts", "compare_symptoms"}
    if tool_name in knowledge_tools and not any(t in field_tools for t in tools_used):
        reward -= 0.04

    return round(reward, 4)


def _count_spam(diag_text: str, actual_problems: set) -> int:
    """Count how many problem names are mentioned that are NOT actual problems."""
    spam = 0
    for name in ALL_PROBLEM_NAMES:
        variants = [name, name.replace("_", " ")]
        if any(v in diag_text for v in variants):
            if name not in actual_problems:
                spam += 1
    return spam


def compute_terminal_score(episode: dict, findings: list, diagnosis_params: str | None,
                           tools_used: list, state) -> float:
    """Terminal reward with anti-exploit measures."""
    problems = episode["problems"]
    ground_truth = episode["ground_truth_treatments"]
    difficulty = episode.get("difficulty", "easy")
    cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["easy"])
    actual_names = {p["name"] for p in problems}

    diag_text = (diagnosis_params or "").lower()

    if not diag_text or len(diag_text) < 10:
        return 0.05

    # 1. Problem identification (only in diagnosis text)
    identified = 0
    for p in problems:
        name_variants = [p["name"], p["name"].replace("_", " ")]
        if any(v in diag_text for v in name_variants):
            identified += 1
    id_score = (identified / len(problems)) * cfg["id_weight"]

    # 2. Treatment quality
    treat_score = 0.0
    for gt in ground_truth:
        treatment = gt.get("treatment", {})
        for key, val in treatment.items():
            if isinstance(val, str):
                val_clean = val.replace("_", " ").lower()
                if val_clean in diag_text:
                    treat_score += cfg["treat_weight"] / len(ground_truth)
                    break

    # 3. Evidence chain — unique relevant tools
    unique_tools = len(set(tools_used))
    evidence_ratio = min(1.0, unique_tools / cfg["evidence_min"])
    evidence_score = evidence_ratio * 0.15

    # 4. Budget efficiency — must spend SOME but not too much
    budget_pct = state.budget_used / episode["budget"]
    if budget_pct < cfg["min_budget_pct"]:
        # Spent too little — didn't investigate properly
        efficiency_score = budget_pct / cfg["min_budget_pct"] * 0.05
    elif budget_pct > 0.7:
        # Spent too much — wasteful
        efficiency_score = max(0, (1.0 - budget_pct) * 0.10)
    else:
        # Sweet spot
        efficiency_score = 0.10

    # 5. Completeness bonus
    completeness = 0.10 if identified == len(problems) else 0.0

    # 6. ANTI-SPAM PENALTY — penalize mentioning problems not supported by evidence
    spam_count = _count_spam(diag_text, actual_names)
    spam_penalty = spam_count * 0.04  # each false mention costs 4%

    # 7. Violation penalty
    violation_penalty = state.violations * 0.10

    # 8. Step penalty for hard
    step_penalty = 0.0
    if difficulty == "hard":
        step_penalty = max(0, (state.step_count - 10) * 0.005)

    total = (id_score + treat_score + evidence_score + efficiency_score
             + completeness - spam_penalty - violation_penalty - step_penalty)
    return round(min(1.0, max(0.0, total)), 4)
