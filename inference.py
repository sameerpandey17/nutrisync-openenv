"""
NutriSync V2 Inference Script  OpenEnv Standard Format
========================================================

Environment Variables:
    OPENAI_API_KEY   Your OpenAI API key (required)
    API_BASE_URL     The LLM endpoint (default: https://api.openai.com/v1)
    MODEL_NAME       The model to use (default: Qwen/Qwen2.5-72B-Instruct)
    NUTRISYNC_TASK   The task to run: easy | medium | hard (default: easy)

Usage:
    set OPENAI_API_KEY=sk-...
    python inference.py

STDOUT Format:
    [START] task=<task> env=nutrisync model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

LOG_FILE = "safe_output.log"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")

def _log_to_file(msg: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

from openai import OpenAI

# ---------------------------------------------------------------------------
# Path setup (so models.py and server/ are importable from root)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))

from server.environment import NutrisyncEnv
from server.tasks import TASKS
from models import NutrisyncAction, IngredientItem

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
IMAGE_NAME    = os.getenv("nutrisync-v2")
API_KEY       = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME     = os.getenv("NUTRISYNC_TASK", "easy")
BENCHMARK     = "nutrisync"

MAX_STEPS     = 4          # NutriSync is always exactly 4 steps (breakfastsnack)
TEMPERATURE   = 0.1
MAX_TOKENS    = 800        # V2 actions include cooking_method
SUCCESS_SCORE_THRESHOLD = 0.5   # normalized score in [0, 1]

# ---------------------------------------------------------------------------
# Logging helpers  emit the exact OpenEnv standard lines
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str, image: Optional[str] = None) -> None:
    parts = [f"task={task}", f"env={env}", f"model={model}"]
    if image:
        parts.append(f"image={image}")
    msg = f"[START] {' '.join(parts)}"
    print(msg, flush=True)
    _log_to_file(msg)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    msg = f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}"
    print(msg, flush=True)
    _log_to_file(msg)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    msg = f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}"
    print(msg, flush=True)
    _log_to_file(msg)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are NutriAgent V2, an expert AI dietitian planning a balanced Indian daily diet.

    You must return a JSON object with a single key "items", each element having:
        - "ingredient": string (MUST be from available_ingredients)
        - "quantity": float (grams)
        - "cooking_method": one of ["raw", "boiled", "steamed", "sauteed", "fried", "roasted", "fermented"]

    *** STRATEGIC CHEAT SHEET (CRITICAL FOR HIGH SCORES) ***
    1. BUDGET PACING: Do NOT front-load expenses. Reserve ~40% for Lunch and ~35% for Dinner, about 15% for Breakfast and 10% for Snacks. A single meal blowing >3% past the total budget triggers a KILL SWITCH (Score=0).
    2. VARIETY BONUS: Use 12+ total UNIQUE ingredients across all meals to earn a massive +0.8 bonus.
    3. ANTI-SPAM (CLUSTER REUSE): Reusing the same cluster (food group) >2 times triggers a 0.70x multiplication penalty.
    4. SATIETY WINDOW: You MUST keep satiety between [40, 80]. Going outside this applies a 0.90x multiplication penalty. Add fiber/protein to increase; reduce simple sugars.
    5. MEAL COMPLETENESS: Every meal MUST have Protein + Carbs + Fat. Missing one applies a 0.85x multiplication penalty.
    6. PROTEIN PACING: If `protein_pace_deficit` is actively constraining your space, `available_ingredients` will ONLY show high-protein items. You must pick from them to avoid penalties.
    7. FATAL GATES: Never exceed the remaining budget, select forbidden items, or ignore diet instructions.

    Other Strategy guidelines:
    - Target Macros: Protein 2035%, Carbs 4055%, Fat 2030%.
    - Minimum Micronutrients: Iron > 15mg, Calcium > 600mg, Fiber > 25g, Vitamin C > 65mg.
    - Calorie Density Limit: Maintain high-density fats/sugars (ghee, oil, butter) below 30% of meal calories.
    - Glycemic Load (GL): Avoid pairing simple sugars with high GI foods without fiber-rich sides.
    - Use ONLY ingredients from available_ingredients.
    - Include protein + grain + fat source in every meal for completeness.
    - Distribute calories: breakfast=25%, lunch=35%, dinner=30%, snack=10% of daily target.
    - Prefer cooking methods appropriate for each ingredient (steamed for spinach, raw for curd, etc.).

    Example response:
    {"items": [
      {"ingredient": "moong_dal", "quantity": 100, "cooking_method": "boiled"},
      {"ingredient": "roti", "quantity": 80, "cooking_method": "roasted"},
      {"ingredient": "spinach", "quantity": 100, "cooking_method": "steamed"}
    ]}

    Return only valid JSON. No additional text.
""").strip()


def build_user_prompt(obs_data: dict, step: int) -> str:
    daily_totals = obs_data.get('daily_totals', {})
    cluster_counts = obs_data.get('cluster_usage_counts', {})
    return textwrap.dedent(f"""
        Step {step}  Plan the {obs_data['current_meal'].upper()}

        Remaining targets:
          Calories left        : {obs_data['calories_left']:.1f} kcal
          Protein left         : {obs_data['protein_left']:.1f} g
          Budget left          : Rs.{obs_data['budget_left']:.2f}
          Protein pace deficit : {obs_data.get('protein_pace_deficit', 0.0):.1f} g behind pace

        V2 State:
          Satiety              : {obs_data.get('satiety', 50.0):.1f} / 100 (target window: 4080)
          Budget gate          : {obs_data.get('budget_gate', 1.0):.1f}
          Hard fail            : {obs_data.get('hard_fail', False)}
          Action constrained   : {obs_data.get('action_space_constrained', False)}
          Cumulative protein   : {obs_data.get('cumulative_protein_g', 0.0):.1f} g
          Cluster usage        : {json.dumps(cluster_counts)}
          Daily micronutrients : iron={daily_totals.get('iron_mg', 0.0):.1f}mg  calcium={daily_totals.get('calcium_mg', 0.0):.0f}mg  fiber={daily_totals.get('fiber_g', 0.0):.1f}g  vitC={daily_totals.get('vitamin_c_mg', 0.0):.1f}mg

        Constraints: {json.dumps(obs_data['constraints'], indent=2)}

        AVAILABLE ingredients (use ONLY these): {', '.join(obs_data.get('available_ingredients', obs_data['allowed_ingredients']))}
        Unavailable this episode: {', '.join(obs_data.get('unavailable_ingredients', []))}

        Return only the JSON action for this meal (include cooking_method per ingredient).
    """).strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI,
    obs_data: dict,
    step: int,
    allowed_ingredients: List[str],
) -> tuple[Optional[NutrisyncAction], str, Optional[str]]:
    """
    Ask the LLM for a meal plan and parse it into a NutrisyncAction.
    Returns (action_or_None, action_str_for_logging, error_or_None).
    """
    user_prompt = build_user_prompt(obs_data, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        action_dict = json.loads(raw)

        # Validate ingredients and extract cooking_method (V2)
        VALID_METHODS = {"raw", "boiled", "steamed", "sauteed", "fried", "roasted", "fermented"}
        validated_items = []
        for item in action_dict.get("items", []):
            ing = item.get("ingredient", "").lower().strip()
            qty = float(item.get("quantity", 0))
            method = item.get("cooking_method", "boiled").lower().strip()
            if method not in VALID_METHODS:
                method = "boiled"   # fallback to safe default
            if ing in allowed_ingredients and qty > 0:
                validated_items.append(
                    IngredientItem(ingredient=ing, quantity=qty, cooking_method=method)
                )

        if not validated_items:
            return None, "[]", "No valid ingredients in model output"

        action = NutrisyncAction(items=validated_items)
        action_str = json.dumps([
            {"ingredient": i.ingredient, "quantity": i.quantity, "cooking_method": i.cooking_method}
            for i in validated_items
        ])
        return action, action_str, None

    except Exception as exc:
        return None, "[]", str(exc)


# ---------------------------------------------------------------------------
# Main run loop  one task at a time
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> float:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Choose from: {list(TASKS.keys())}")

    config  = TASKS[task_id].copy()
    grader  = config.pop("grader")
    env     = NutrisyncEnv(**config, seed=42)
    obs     = env.reset()

    rewards:    List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME, image=IMAGE_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            obs_data = {
                "current_meal":           obs.current_meal,
                "calories_left":          obs.calories_left,
                "protein_left":           obs.protein_left,
                "budget_left":            obs.budget_left,
                "allowed_ingredients":    obs.allowed_ingredients,
                "available_ingredients":  obs.available_ingredients or obs.allowed_ingredients,
                "unavailable_ingredients": obs.unavailable_ingredients,
                "constraints":            obs.constraints,
                # V2 fields
                "satiety":                obs.satiety,
                "protein_pace_deficit":   obs.protein_pace_deficit,
                "budget_gate":            obs.budget_gate,
                "action_space_constrained": obs.action_space_constrained,
                "cluster_usage_counts":   obs.cluster_usage_counts,
                "daily_totals":           obs.daily_totals,
                "hard_fail":              getattr(obs, 'hard_fail', False),
                "cumulative_protein_g":   getattr(obs, 'cumulative_protein_g', 0.0),
            }

            # Pass allowed but let environment handle availability penalization if the model disobeys
            action, action_str, llm_error = get_agent_action(
                client, obs_data, step, obs.allowed_ingredients
            )

            if action is None:
                # Use empty meal to avoid crashing the environment
                action = NutrisyncAction(items=[])

            obs, reward, done, info = env.step(action)
            step_error = info.get("error") or llm_error
            rewards.append(reward.score)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward.score,
                done=done,
                error=step_error,
            )

            if done:
                break

        # Final grade via the task's programmatic grader
        final_state = env.get_state()
        score   = grader(final_state)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print(
            "[ERROR] No API key found.\n"
            "Set OPENAI_API_KEY environment variable.\n"
            "Example: set OPENAI_API_KEY=sk-your_key_here",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Run all three tasks if NUTRISYNC_TASK is not set, else just the one
    tasks_to_run = (
        [TASK_NAME] if os.getenv("NUTRISYNC_TASK") else ["easy", "medium", "hard"]
    )

    all_scores = {}
    for task_id in tasks_to_run:
        print(f"\n{'='*50}", flush=True)
        score = run_task(client, task_id)
        all_scores[task_id] = score
        print(f"{'='*50}", flush=True)

    if len(all_scores) > 1:
        avg = sum(all_scores.values()) / len(all_scores)
        
        msg = f"\n[SUMMARY] Average Score: {avg:.3f}/1\n"
        for t, s in all_scores.items():
            msg += f"  {t:<8}: {s:.3f}/1\n"
        
        print(msg, flush=True)
        _log_to_file(msg)


if __name__ == "__main__":
    main()
