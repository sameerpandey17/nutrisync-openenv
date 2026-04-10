---
title: NutriSync V2
emoji: 🥗
colorFrom: teal
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - nutrition
  - evaluation
---

# NutriSync V2

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment for multi-step dietary planning. Each episode requires an agent to plan 4 sequential meals — Breakfast, Lunch, Dinner, and Snack — satisfying hard nutritional and budget constraints. Scores are returned in **[0, 1]**.

The environment is designed to be **non-exploitable**: a multiplicative reward pipeline with kill-switches, cluster spam detection, and persistent satiety state ensures that only genuinely balanced, diverse meal plans score well.

---

## Quick Start

```python
from server.environment import NutrisyncEnv
from server.tasks import TASKS
from models import NutrisyncAction, IngredientItem

# Load a task configuration
cfg = TASKS["easy"].copy()
grader = cfg.pop("grader")

env = NutrisyncEnv(**cfg, seed=42)
obs = env.reset()

# Plan breakfast
action = NutrisyncAction(items=[
    IngredientItem(ingredient="oats",    quantity=80,  cooking_method="boiled"),
    IngredientItem(ingredient="banana",  quantity=100, cooking_method="raw"),
    IngredientItem(ingredient="milk",    quantity=150, cooking_method="boiled"),
    IngredientItem(ingredient="ghee",    quantity=10,  cooking_method="raw"),
])
obs, reward, done, info = env.step(action)
print(f"Step reward: {reward.score:.4f}")

# ... plan lunch, dinner, snack ...

state = env.get_state()
final_score = grader(state)   # float in [0, 1]
print(f"Episode score: {final_score:.4f}")
```

---

## Running the Inference Agent

The `inference.py` script runs a fully automated LLM agent (via [Groq](https://groq.com) or any OpenAI-compatible endpoint) through all three tasks.

```bash
# Set your Groq API key
set GROQ_API_KEY=gsk_...

# Run all three tasks
python inference.py

# Run a specific task
set NUTRISYNC_TASK=easy
python inference.py
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Primary API key (Groq) |
| `HF_TOKEN` | — | Fallback for HuggingFace router |
| `OPENAI_API_KEY` | — | Fallback for OpenAI |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | LLM endpoint |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Model identifier |
| `NUTRISYNC_TASK` | *(all three)* | `easy`, `medium`, or `hard` |

**STDOUT format:**

```
[START] task=easy env=nutrisync model=llama-3.3-70b-versatile
[STEP]  step=1 action=[...] reward=0.2341 done=false error=null
[STEP]  step=2 action=[...] reward=0.1892 done=false error=null
[STEP]  step=3 action=[...] reward=0.1450 done=false error=null
[STEP]  step=4 action=[...] reward=0.5912 done=true  error=null
[END]   success=true steps=4 score=0.7490 rewards=0.23,0.19,0.14,0.59
```

---

## Running the Interactive UI

A Gradio interface is included for manual exploration:

```bash
python app.py
# Open http://localhost:7860
```

Features:
- Difficulty selector and custom budget override
- Allergy / restriction checkbox panel (blocks selected ingredients)
- 4-slot meal builder with ingredient, quantity, and cooking method per slot
- Live reward signal, satiety, cluster usage, and gate warnings after each meal
- Episode summary with score bar on completion

---

## Reward System

Scores are computed in **[0, 1]** by `compute_episode_reward()` in `server/reward.py`.

The pipeline is **Build-From-Zero**: points accumulate additively, then multiplicative penalties shrink the total. Kill-switches short-circuit to 0 immediately.

### Kill Switches (instant score = 0)

| Trigger | Condition |
|---|---|
| `hard_fail` | Agent used an allergen, violated diet mode (e.g. meat in vegetarian), or exceeded per-ingredient quota |
| `budget_gate = 0` | Any single meal caused cumulative spend to exceed budget by more than 3% |

### Budget Gate Warning

If a meal causes spend to exceed 0–3% over budget, `budget_gate` is set to `0.2` (warning state). The entire score is multiplied by `0.2` at the end — a near-fatal penalty that is **irreversible**.

---

### Additive Bonuses (build the score)

These are summed before any multiplicative penalties are applied. The theoretical maximum base is ~11.0, normalized to [0, 1] by dividing by 10.

| Component | Max | How Earned |
|---|---|---|
| Calorie pacing | +0.20 | +0.05/meal if within 5% of target fraction; +0.025 if within 10% |
| Macro ratio (protein) | +0.10 | Protein calories 20–35% of daily total |
| Macro ratio (carbs) | +0.10 | Carb calories 40–55% of daily total |
| Macro ratio (fat) | +0.10 | Fat calories 20–30% of daily total |
| Iron floor | +0.05 | Daily iron >= 15 mg |
| Calcium floor | +0.05 | Daily calcium >= 600 mg |
| Fiber floor | +0.05 | Daily fiber >= 25 g |
| Vitamin C floor | +0.05 | Daily vitamin C >= 65 mg |
| Cooking coherence | +0.12 | +0.03/meal if >= 60% of ingredients use their optimal cooking method |
| Ingredient variety | +0.08 | >= 12 unique ingredients AND >= 4 clusters across all meals |
| Ingredient variety (partial) | +0.04 | >= 8 unique ingredients AND >= 3 clusters |
| Budget efficiency (optimal) | +0.07 | 85–95% of budget used |
| Budget efficiency (good) | +0.03 | 70–85% of budget used |
| Clean run bonus | +0.30 | Zero availability violations across all meals |

> **Calorie pacing target fractions:** Breakfast 25%, Lunch 35%, Dinner 30%, Snack 10% of daily target.

---

### Multiplicative Penalties (shrink the score)

Applied after the additive phase, in order. Each penalty compounds with the others.

| Penalty | Multiplier | Trigger |
|---|---|---|
| Cluster spam | `x 0.85` per item over limit | Ingredient cluster used more than its limit (see table below) |
| Satiety out of window | `x 0.95` per missed step | Satiety outside [40, 80] at the start of a step (step 2 onward) |
| High glycemic load | `x 0.98` per 0.20 of penalty | GL > 20 per meal = -0.50; GL 10–20 = -0.20 (breakfast + snack, or all meals in hard) |
| Incomplete meal | `x 0.95` per incomplete meal | Meal missing protein, carb, or fat cluster |
| Availability violation | `x 0.75` per violation | Unavailable (seasonally absent) ingredient selected |

**Cluster spam limits:**

| Cluster | Limit |
|---|---|
| vegetables | 8 |
| fats | 6 |
| whole_grains | 6 |
| proteins | 4 |
| dairy | 4 |
| legumes | 4 |
| refined_carbs | 3 |

---

### Satiety Dynamics

Satiety starts at 50 and updates after each meal:

```
delta = (fiber_g × 0.8) + (protein_g × 0.4) - (simple_sugar_g × 0.6)
satiety = clamp(satiety + delta, 0, 100)
```

A satiety penalty fires at the **start** of each step (from step 2 onward) if satiety is outside [40, 80].

---

### Per-Step Dense Reward (UI Feedback Only)

The `RewardEngine.compute()` method provides per-meal feedback for the UI. This is **not** the episode grader score. It consists of:

**Tier 1 — Per-meal scores (positive):**

| Component | Max | Notes |
|---|---|---|
| `calorie_proximity` | 1.0 | Distance from meal's target calorie fraction |
| `protein_proximity` | 1.0 | Distance from meal protein range (medium/hard only) |
| `cost_pacing` | 1.0 | Distance from expected cost fraction per meal |
| `ingredient_diversity` | 1.0 | 1 ingredient = 0.2, 2 = 0.5, 3 = 0.8, 4+ = 1.0 |
| `food_group_coverage` | 1.0 | 1 group = 0.3, 2 = 0.7, 3+ = 1.0 |
| `portion_realism` | 1.0 | Fraction of ingredients in realistic gram range |

**Tier 2 — Anti-gaming penalties (negative):**

| Component | Range | Notes |
|---|---|---|
| `repetition_penalty` | 0 to -0.6 | -0.15 per ingredient reused from a previous meal |
| `empty_meal_penalty` | -0.5 | Flat penalty if meal has zero valid items |
| `calorie_dense_penalty` | -0.15 / -0.30 | High-density fats/sugars > 30% or > 50% of meal calories |
| `satiety_state` | -0.3 | Satiety outside [40, 80] this step |
| `budget_gate_signal` | -1.0 / -2.0 | Budget gate in warning or blown state |
| `hard_fail_signal` | -5.0 | Any hard constraint violated |

**Tier 3 — Episode completion bonuses (final step only):**

| Component | Max | Notes |
|---|---|---|
| `final_calorie_accuracy` | 3.0 | <= 5% daily deviation = 3.0; <= 10% = 2.0; <= 20% = 1.0 |
| `final_protein_accuracy` | 2.0 | <= 5% deviation = 2.0; <= 10% = 1.5; <= 20% = 0.8 |
| `budget_utilization` | 1.0 | 70–95% used = 1.0; 95–100% = 0.7; 50–70% = 0.5 |
| `global_diversity` | 2.0 | >= 12 ingredients = 2.0; >= 10 = 1.5; >= 7 = 1.0; >= 4 = 0.5 |
| `meal_balance` | 2.0 | `2 × (1 - avg_deviation)` from ideal meal fractions |
| `cross_meal_variety` | 0 to -1.5 | Ingredient in >= 4 meals = -0.5; >= 3 meals = -0.3 per such ingredient |

---

## Tasks

Three fixed configurations are defined in `server/tasks.py`:

| Task | Diet | Budget | Calories | Protein | GL Penalty | Coherence | Unavailable |
|---|---|---|---|---|---|---|---|
| `easy` | Omnivore | Rs. 250 | 2000 kcal | 100 g | Disabled | Disabled | 8 |
| `medium` | Vegetarian | Rs. 200 | 2300 kcal | 100 g | Breakfast + Snack | Disabled | 10 |
| `hard` | Vegan | Rs. 160 | 2300 kcal | 120 g | All 4 meals | Enabled | 12 |

Hard task also enforces `allergies: [coconut, coconut_oil]` and usage limits `{rice: 2, mustard_oil: 2}`.

Seasonal unavailability is sampled randomly at each `reset()` using the episode seed.

---

## Episode Structure

Each episode is exactly **4 steps**:

1. `reset()` returns the initial observation (current meal = `breakfast`)
2. `step(action)` submits a `NutrisyncAction` for the current meal, returns next observation + per-step reward
3. Steps repeat for `lunch`, `dinner`, `snack`
4. After step 4, `done = True` — call the task grader for the final [0, 1] score

```python
state = env.get_state()
score = grader(state)   # float in [0, 1]
```

### Action

**NutrisyncAction** — a list of ingredients for the current meal:

```python
NutrisyncAction(items=[
    IngredientItem(ingredient="chicken",  quantity=150.0, cooking_method="boiled"),
    IngredientItem(ingredient="rice",     quantity=150.0, cooking_method="boiled"),
    IngredientItem(ingredient="spinach",  quantity=100.0, cooking_method="steamed"),
    IngredientItem(ingredient="mustard_oil", quantity=10.0, cooking_method="raw"),
])
```

- `ingredient`: must be a key in the 50-item database (see `server/environment.py`)
- `quantity`: grams (float)
- `cooking_method`: one of `raw`, `boiled`, `steamed`, `sauteed`, `roasted`, `fried`, `fermented`

Cooking methods apply nutritional modifiers to calories, protein, vitamin C, and fiber (e.g. `fried` adds +45% calories, `fermented` boosts protein +5% and vitamin C +10%).

### Observation

**NutrisyncObservation** — returned after each step:

```python
obs.current_meal            # "breakfast" | "lunch" | "dinner" | "snack" | "done"
obs.calories_left           # float — calories remaining to target
obs.protein_left            # float — protein remaining to target
obs.budget_left             # float — Rs. remaining
obs.satiety                 # float [0, 100] — target window [40, 80]
obs.budget_gate             # 1.0 (ok) | 0.2 (warning) | 0.0 (blown)
obs.hard_fail               # bool — True if a fatal constraint was violated
obs.action_space_constrained  # bool — True if protein pace forces high-protein selection
obs.available_ingredients   # List[str] — filtered by diet, availability, protein pace
obs.unavailable_ingredients # List[str] — seasonally absent this episode
obs.cluster_usage_counts    # Dict[str, int] — spam tracking
obs.daily_totals            # Dict with iron_mg, calcium_mg, fiber_g, vitamin_c_mg
obs.protein_pace_deficit    # float — grams behind expected protein pace
```

---

## Ingredient Database

50 ingredients across 10 clusters (per 100 g cooked):

| Cluster | Ingredients |
|---|---|
| proteins | chicken, mutton, fish_rohu, eggs, tofu |
| legumes | moong_dal, toor_dal, chana_dal, masoor_dal, rajma, chickpeas, paneer |
| whole_grains | wheat_flour, roti, oats, millet_bajra, ragi |
| refined_carbs | rice, poha, semolina, bread |
| vegetables | potato, onion, tomato, cauliflower, cabbage, brinjal, lady_finger, bottle_gourd, carrot, green_peas, capsicum, coconut |
| leafy_greens | spinach |
| fruits | banana, apple, mango, papaya, guava |
| dairy | milk, curd, paneer, cheese |
| fats | ghee, butter, mustard_oil, coconut_oil, groundnut_oil |
| simple_sugars | sugar, jaggery, honey |

---

## Project Structure

```
NutriSync/
├── README.md               # This file
├── openenv.yaml            # OpenEnv manifest
├── Dockerfile              # Container image
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Minimal runtime requirements
├── models.py               # Pydantic models (Action, Observation, State, Reward)
├── inference.py            # Automated LLM agent runner (Groq / OpenAI-compatible)
├── app.py                  # Gradio interactive UI
├── test_v2.py              # Integration tests
├── deploy_hf.py            # Hugging Face Spaces upload helper
├── .env.example            # Environment variable template
└── server/
    ├── __init__.py
    ├── environment.py      # NutrisyncEnv — core V2 environment + nutrition DB
    ├── reward.py           # RewardEngine + compute_episode_reward()
    ├── tasks.py            # Three task configs + programmatic graders
    └── app.py              # FastAPI server + Gradio mount (unified deployment)
```

---

## Deploying to Hugging Face Spaces

```bash
set HF_TOKEN=hf_...
python deploy_hf.py
```

The script uploads all source files to `strangesam17/nutrisync-v2` (Space), excluding `.git`, `.venv`, `.env*`, `__pycache__`, and log files.

After deployment the Space exposes:

- `/gradio` — Interactive Gradio UI
- `/reset`, `/step`, `/health` — OpenEnv HTTP API
- `/docs` — Swagger / OpenAPI documentation

---

## Learn More

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv Environment Design Guide](https://github.com/meta-pytorch/OpenEnv/blob/main/README.md)
- [Groq API](https://console.groq.com)
- [HuggingFace Spaces](https://huggingface.co/spaces/strangesam17/nutrisync-v2)
