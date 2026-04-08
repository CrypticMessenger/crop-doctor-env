"""CropDoctorEnv inference script with mandatory [START]/[STEP]/[END] log format."""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import CropDoctorEnv
from models import CropAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "crop_doctor_env"
TEMPERATURE = 0.0
MAX_TOKENS = 300
SEED = 42

TASKS = [
    {"task_id": "easy", "max_steps": 15},
    {"task_id": "medium", "max_steps": 20},
    {"task_id": "hard", "max_steps": 25},
]

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert agricultural scientist diagnosing crop problems in Indian farms.
You have access to diagnostic tools. Each turn, choose ONE tool to use.

Reply with EXACTLY this JSON format (no other text):
{"tool": "<tool_name>", "parameters": "<optional details>"}

When you have enough evidence, use:
{"tool": "submit_diagnosis", "parameters": "<your diagnosis and recommended treatment>"}

Strategy: Start with field inspections, then targeted tests based on symptoms.
Be efficient — you have limited budget and time.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_prompt(obs_dict: dict, history: list) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    tools_list = "\n".join(f"  - {t}" for t in obs_dict.get("available_tools", []))
    return textwrap.dedent(f"""
Current situation:
  {obs_dict.get('crop_info', '')}
  Budget remaining: ₹{obs_dict.get('budget_remaining', 0)}
  Days remaining: {obs_dict.get('days_remaining', 0)}
  Lab slots remaining: {obs_dict.get('lab_slots_remaining', 0)}
  Step: {obs_dict.get('step_number', 0)}

Last result: {obs_dict.get('tool_result', 'N/A')}

Findings so far:
{obs_dict.get('findings_so_far', 'None')}

Available tools:
{tools_list}

Previous actions:
{history_block}

Choose your next tool (JSON format):
""").strip()


def get_action(client: OpenAI, obs_dict: dict, history: list) -> tuple:
    prompt = build_prompt(obs_dict, history)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Robust JSON extraction
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        # Handle multi-line values by replacing newlines in JSON string values
        import re
        text = re.sub(r'(?<=: ")(.*?)(?=")', lambda m: m.group(0).replace('\n', ' '), text, flags=re.DOTALL)
        # Try to find JSON object in text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        parsed = json.loads(text)
        return parsed.get("tool", "inspect_leaves"), str(parsed.get("parameters", ""))
    except Exception:
        return "inspect_leaves", ""


async def run_task(task_id: str, max_steps: int) -> float:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = None
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        if IMAGE_NAME:
            env = await CropDoctorEnv.from_docker_image(IMAGE_NAME)
        else:
            base_url = os.getenv("HF_SPACE_URL", "http://localhost:7860")
            env = CropDoctorEnv(base_url=base_url)
            await env.__aenter__()

        history = []
        result = await env.reset(task_id=task_id)
        obs_dict = result.observation.model_dump() if hasattr(result.observation, 'model_dump') else {}

        for step in range(1, max_steps + 1):
            if result.done:
                break

            tool, params = get_action(client, obs_dict, history)
            result = await env.step(CropAction(tool=tool, parameters=params))

            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step

            obs_dict = result.observation.model_dump() if hasattr(result.observation, 'model_dump') else {}
            log_step(step=step, action=f"{tool}({params})" if params else tool, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {tool} -> reward {reward:+.2f}")

            if done:
                break

        terminal = rewards[-1] if rewards else 0.0
        score = max(0.0, min(1.0, terminal))
        success = score >= 0.3

    except Exception as e:
        print(f"[DEBUG] error: {e}", flush=True)

    finally:
        if env:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main():
    for task in TASKS:
        await run_task(task["task_id"], task["max_steps"])


if __name__ == "__main__":
    asyncio.run(main())
