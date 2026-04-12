"""CropDoctorEnvironment — the core environment logic."""

import json, uuid
from typing import Optional
from openenv.core.env_server import Environment

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import CropAction, CropObservation, CropState
from server.generator import generate_episode
from server.tools import TOOL_REGISTRY, TOOL_DESCRIPTIONS
from server.graders import compute_step_reward, compute_terminal_score

TASKS = {"easy": "easy", "medium": "medium", "hard": "hard"}


class CropDoctorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = CropState()
        self._episode = {}
        self._tools_used = []
        self._findings = []
        self._diagnosis_submitted = False

    def reset(self, seed=None, episode_id=None, task_id="easy", **kwargs) -> CropObservation:
        difficulty = TASKS.get(task_id, "easy")
        self._episode = generate_episode(difficulty, seed=seed)
        self._episode["episode_id"] = episode_id or self._episode["episode_id"]
        self._tools_used = []
        self._findings = []
        self._diagnosis_submitted = False

        ep = self._episode
        self._state = CropState(
            episode_id=ep["episode_id"],
            step_count=0,
            task_id=task_id,
            difficulty=difficulty,
            crop=ep["crop"],
            growth_stage=ep["growth_stage"],
            soil_type=ep["soil_type"],
            problems=json.dumps(ep["problems"]),
            findings="[]",
            budget_used=0,
            days_used=0.0,
            lab_slots_used=0,
            violations=0,
        )

        return CropObservation(
            done=False,
            reward=None,
            crop_info=f"Crop: {ep['crop']} | Stage: {ep['growth_stage']} | Soil: {ep['soil_type']} | Season: {ep['crops_db'][ep['crop']].get('season', 'unknown')}",
            tool_result="A farmer has reported problems with their crop. Begin your investigation.",
            findings_so_far="No findings yet.",
            available_tools=list(TOOL_DESCRIPTIONS.keys()),
            budget_remaining=ep["budget"],
            days_remaining=ep["days"],
            lab_slots_remaining=ep["lab_slots"],
            step_number=0,
            message="Episode started. Use tools to investigate the crop problem and submit a diagnosis.",
        )

    def step(self, action: CropAction, **kwargs) -> CropObservation:
        ep = self._episode
        if not ep:
            return CropObservation(
                done=True, reward=0.01,
                crop_info="", tool_result="Error: call reset() before step().",
                findings_so_far="", available_tools=[],
                budget_remaining=0, days_remaining=0, lab_slots_remaining=0,
                step_number=0, message="Must call reset() first.",
            )
        self._state.step_count += 1
        tool_name = action.tool.strip()

        # Check if episode already done
        if self._diagnosis_submitted:
            return self._make_obs(0.01, True, "Episode already ended.", "Diagnosis was already submitted.")

        # Check tool exists
        if tool_name not in TOOL_REGISTRY and tool_name != "submit_diagnosis":
            self._state.violations += 1
            return self._make_obs(0.05, False, f"Unknown tool: {tool_name}", f"Tool '{tool_name}' does not exist. Use one of the available tools.")

        # Check max steps
        if self._state.step_count > ep["max_steps"]:
            self._diagnosis_submitted = True
            score = compute_terminal_score(ep, self._findings, None, self._tools_used, self._state)
            return self._make_obs(score, True, "Max steps reached. Episode ended.", "Ran out of steps.")

        # Handle submit_diagnosis
        if tool_name == "submit_diagnosis":
            return self._handle_diagnosis(action.parameters)

        reg = TOOL_REGISTRY[tool_name]

        # Check dependencies
        violation = self._check_dependencies(tool_name, reg)
        if violation:
            self._state.violations += 1
            return self._make_obs(0.01, False, violation, f"Rule violation: {violation}")

        # Check resources
        fn = reg["fn"]
        result_text, cost, days = fn(ep)

        if reg["category"] == "lab":
            if self._state.lab_slots_used >= ep["lab_slots"]:
                return self._make_obs(0.10, False, "No lab slots remaining.", "All 3 lab slots used.")
            self._state.lab_slots_used += 1

        if self._state.budget_used + cost > ep["budget"]:
            return self._make_obs(0.10, False, "Insufficient budget.", f"Need ₹{cost} but only ₹{ep['budget'] - self._state.budget_used} left.")

        if self._state.days_used + days > ep["days"]:
            return self._make_obs(0.10, False, "Not enough field days.", f"Need {days}d but only {ep['days'] - self._state.days_used:.1f}d left.")

        # Execute tool
        self._state.budget_used += cost
        self._state.days_used += days
        self._tools_used.append(tool_name)

        # Track findings
        finding = {"tool": tool_name, "result": result_text, "step": self._state.step_count}
        self._findings.append(finding)
        self._state.findings = json.dumps([f["tool"] for f in self._findings])

        # Compute step reward
        reward = compute_step_reward(tool_name, result_text, ep, self._tools_used)

        return self._make_obs(reward, False, result_text, f"Tool '{tool_name}' executed. Cost: ₹{cost}, Time: {days}d")

    def _handle_diagnosis(self, parameters: str) -> CropObservation:
        ep = self._episode
        # Check prerequisites
        inspections = [t for t in self._tools_used if t.startswith("inspect_") or t == "check_pest_presence"]
        # Require a PAID test (not test_soil_type which is free)
        paid_tests = {"test_soil_ph", "test_soil_npk", "test_soil_micronutrients", "test_soil_moisture",
                      "test_water_quality", "send_leaf_sample", "send_soil_sample", "identify_pest_species"}
        tests = [t for t in self._tools_used if t in paid_tests]
        if not inspections or not tests:
            self._state.violations += 1
            return self._make_obs(0.01, False,
                "Cannot submit diagnosis without at least 1 field inspection and 1 test.",
                "Rule violation: insufficient evidence.")

        self._diagnosis_submitted = True
        score = compute_terminal_score(ep, self._findings, parameters, self._tools_used, self._state)
        return self._make_obs(score, True,
            f"Diagnosis submitted. Final score: {score:.3f}",
            "Episode complete.")

    def _check_dependencies(self, tool_name: str, reg: dict) -> Optional[str]:
        needs = reg.get("needs_prior", [])
        for dep in needs:
            if dep == "_any_inspection":
                if not any(t.startswith("inspect_") or t == "check_pest_presence" for t in self._tools_used):
                    return f"Cannot use {tool_name} without at least one field inspection first."
            elif dep == "_any_test":
                if not any(t.startswith("test_") or t.startswith("send_") for t in self._tools_used):
                    return f"Cannot use {tool_name} without at least one test first."
            elif dep not in self._tools_used:
                return f"Cannot use {tool_name} without running {dep} first."
        return None

    def _make_obs(self, reward: float, done: bool, tool_result: str, message: str) -> CropObservation:
        ep = self._episode
        clamped = round(min(0.99, max(0.01, reward)), 4) if done else round(reward, 4)
        if not ep:
            return CropObservation(done=done, reward=clamped, crop_info="",
                tool_result=tool_result, findings_so_far="", available_tools=[],
                budget_remaining=0, days_remaining=0, lab_slots_remaining=0,
                step_number=0, message=message)
        findings_text = "\n".join(f"- [{f['tool']}]: {f['result'][:100]}" for f in self._findings[-5:]) or "No findings yet."
        steps_left = ep.get("max_steps", 15) - self._state.step_count
        if steps_left <= 3 and not done:
            message += f" ⚠️ WARNING: Only {steps_left} steps remaining! Consider submitting your diagnosis."
        return CropObservation(
            done=done,
            reward=clamped,
            crop_info=f"Crop: {ep['crop']} | Stage: {ep['growth_stage']} | Soil: {ep['soil_type']}",
            tool_result=tool_result,
            findings_so_far=findings_text,
            available_tools=list(TOOL_DESCRIPTIONS.keys()),
            budget_remaining=ep["budget"] - self._state.budget_used,
            days_remaining=round(ep["days"] - self._state.days_used, 1),
            lab_slots_remaining=ep["lab_slots"] - self._state.lab_slots_used,
            step_number=self._state.step_count,
            message=message,
        )

    @property
    def state(self) -> CropState:
        return self._state
