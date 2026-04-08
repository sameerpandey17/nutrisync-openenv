---
title: NutriSync V2
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

#  NutriSync V2: Strict RL Diet Planning Environment
A highly sophisticated, strict reinforcement learning environment for sequential meal planning with constraint satisfaction. Built for researchers testing AI agent conformity, foresight, and constrained optimization. Fully compliant with the Meta OpenEnv Specification.

##  Overview

**NutriSync** is a multi-step decision environment where an agent acts as a dietary planner to construct a **4-meal daily diet plan**:
- Breakfast
- Lunch
- Dinner
- Snack

The agent plans **one meal per step** (a fixed 4-step horizon) while satisfying a variety of tight nutritional, dietary, and financial constraints based on real-world scenarios.

### V2 Environment Features

- **Strict "Build-From-Zero" Reward Engine**: Agents start at a score of 0.0. Every point must be explicitly earned by perfectly matching strict macro-ratios, precise calorie pacing per meal, and hitting specific micronutrient floors.
- **Aggressive Multiplicative Penalties**: Additive slaps-on-the-wrist are gone. V2 punishes exploits heavily via multiplicative stacking logic (e.g. going 3% over budget sets your entire multiplier to 0).
- **Persistent State Tracking**: Real-time tracking of cumulative **Satiety**, keeping the agent inside a strict `[40, 80]` window, enforced by punishing low-fiber or high simple-sugar meals.
- **50-Ingredient Database**: Augmented database featuring Glycemic Index (GI), Fiber, Simple Sugars, Iron, Calcium, and Vitamin C.
- **Cooking Methods**: Agents must specify how food is prepared (e.g. `raw`, `steamed`, `fried`), which directly alters the nutritional values of the ingredients.

---

##  Defined Tasks

NutriSync implements three rigorous programmatic grading tasks for deterministic evaluation:

| Task ID | Mode | Description | Key Constraints |
|---------|------|-------------|-----------------|
| `easy` | **Basic Meal Plan** | Simple calorie targeting with a tight budget. | 2000 cal, 250 limit, omnivore |
| `medium` | **Balanced Nutrition** | Requires precise protein targeting and diverse vegetarian sourcing. | 2000 cal, 100g protein, 200 limit |
| `hard` | **Constrained Expert** | Highly constrained vegan planning with allergies and usage limits. | 2000 cal, 120g protein, 160 limit, vegan |

---

##  Action & Observation Space

### 1. Action Space
Agents respond with a `NutrisyncAction` containing an array of ingredient, quantity, and cooking_method JSON dictionaries:

```json
{
  "items": [
    {"ingredient": "rice", "quantity": 100.0, "cooking_method": "boiled"},
    {"ingredient": "dal", "quantity": 150.0, "cooking_method": "sauteed"}
  ]
}
```

### 2. Observation Space
The `NutrisyncObservation` exposes real-time trajectory details so agents can navigate the strict state constraints:

```json
{
  "current_meal": "lunch",
  "calories_left": 1550.0,
  "protein_left": 80.0,
  "budget_left": 120.0,
  "satiety": 65.5,
  "budget_gate": 1.0,
  "hard_fail": false,
  "available_ingredients": ["rice", "dal", "spinach"...],
  "meals_built": {
     "breakfast": [{"ingredient": "idli", "quantity": 200, "cooking_method": "steamed"}]
  },
  "done": false
}
```

---

##  Installation & Usage

### Local Setup
```bash
# Clone the repository
git clone https://github.com/your-username/nutrisync.git
cd nutrisync

# Install dependencies using uv (recommended) or pip
uv sync
pip install openenv-core[core] uvicorn
```

### Baseline LLM Evaluation (OpenAI or HuggingFace)
NutriSync uses a standard `inference.py` that supports **OpenAI API** or **HuggingFace Router**.

1. **Configure Token**: In your PowerShell/Terminal:
```bash
export OPENAI_API_KEY="sk-..."
# OR for HuggingFace:
# export HF_TOKEN="hf_..."
# export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
```

2. **Run Evaluation**:
```bash
python inference.py
```
*Evaluates the LLM across the selected difficulty track using the standard [START]/[STEP]/[END] format.*

### Interactive UI
To manually play with the V2 engine constraints and test meal combinations:
```bash
python app_ui.py
# Available at http://localhost:7860
```

### Extending and Using Locally
You can import the OpenEnv-wrapped environment natively in Python to interact programmatically:

```python
from server.NutriSync_environment import NutrisyncEnvironment
from models import NutrisyncAction, IngredientItem

env = NutrisyncEnvironment()
obs = env.reset(task_id="hard")

action = NutrisyncAction(items=[
    IngredientItem(ingredient="tofu", quantity=150, cooking_method="sauteed")
])

obs = env.step(action)
print(f"Satiety: {obs.satiety}")
```

---

##  License
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the BSD-style license.
