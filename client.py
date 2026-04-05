from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import CropAction, CropObservation, CropState


class CropDoctorEnv(EnvClient[CropAction, CropObservation, CropState]):

    def _step_payload(self, action: CropAction) -> dict:
        return {"tool": action.tool, "parameters": action.parameters}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=CropObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                crop_info=obs_data.get("crop_info", ""),
                tool_result=obs_data.get("tool_result", ""),
                findings_so_far=obs_data.get("findings_so_far", ""),
                available_tools=obs_data.get("available_tools", []),
                budget_remaining=obs_data.get("budget_remaining", 0),
                days_remaining=obs_data.get("days_remaining", 0),
                lab_slots_remaining=obs_data.get("lab_slots_remaining", 0),
                step_number=obs_data.get("step_number", 0),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> CropState:
        return CropState(**{k: v for k, v in payload.items() if k in CropState.model_fields})
