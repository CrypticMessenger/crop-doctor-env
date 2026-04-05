from typing import List, Optional
from openenv.core.env_server import Action, Observation, State


class CropAction(Action):
    tool: str          # Tool name e.g. "inspect_leaves", "test_soil_npk", "submit_diagnosis"
    parameters: str    # JSON string with tool-specific params e.g. '{"gene":"ERBB2"}' style


class CropObservation(Observation):
    # done: bool and reward: Optional[float] inherited from Observation
    crop_info: str                  # Basic crop info visible to agent
    tool_result: str                # Result of last tool used
    findings_so_far: str            # Running summary of discoveries
    available_tools: List[str]      # Tools the agent can use next
    budget_remaining: int           # Rupees left
    days_remaining: float           # Field days left
    lab_slots_remaining: int        # Lab test slots left
    step_number: int
    message: str                    # Feedback message


class CropState(State):
    # episode_id and step_count inherited from State
    task_id: str = ""
    difficulty: str = ""
    crop: str = ""
    growth_stage: str = ""
    soil_type: str = ""
    problems: str = ""             # JSON string of hidden ground truth
    findings: str = ""             # JSON string of agent's discoveries
    budget_used: int = 0
    days_used: float = 0.0
    lab_slots_used: int = 0
    violations: int = 0
