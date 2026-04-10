import sys
import os
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))

from server.environment import NutrisyncEnv, NUTRITION_DATABASE
from models import NutrisyncAction, IngredientItem
from server.tasks import TASKS

# ─── Constants ────────────────────────────────────────────────────────────────
current_env    = None
VALID_METHODS  = ["boiled", "steamed", "raw", "sauteed", "roasted", "fried", "fermented"]
ALL_INGREDIENTS = sorted(NUTRITION_DATABASE.keys())

# ─── Custom CSS ───────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    box-sizing: border-box;
}

/* Background */
body, .gradio-container {
    background: #0d1117 !important;
    min-height: 100vh;
}

/* ── Header card ── */
.ns-header {
    background: linear-gradient(135deg, #0f2942 0%, #0a3d62 60%, #0c5460 100%);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 12px;
    padding: 28px 32px 24px;
    margin-bottom: 18px;
}
.ns-header h1 {
    color: #f0f6fc !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    margin: 0 0 6px 0 !important;
}
.ns-header p {
    color: #8b949e !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}

/* ── Section cards ── */
.ns-card {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 18px !important;
    margin-bottom: 14px;
}

/* ── Section headings ── */
.ns-label {
    color: #38bdf8 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    margin-bottom: 14px !important;
}

/* ── Stat textboxes ── */
.prog-txt textarea, .prog-txt input {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
    text-align: center !important;
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── General inputs ── */
textarea, input[type="number"], input[type="text"], select {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-size: 0.875rem !important;
    transition: border-color 0.15s ease !important;
}
textarea:focus, input:focus, select:focus {
    border-color: #38bdf8 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.12) !important;
}

/* ── Labels ── */
label span, .block label > span {
    color: #8b949e !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* ── Dropdowns ── */
.svelte-select, .wrap {
    background: #0d1117 !important;
    border-color: #30363d !important;
    color: #c9d1d9 !important;
}

/* ── Markdown text ── */
.ns-md p, .ns-md li, .ns-md span { color: #8b949e !important; font-size: 0.875rem !important; }
.ns-md strong { color: #38bdf8 !important; font-weight: 600 !important; }
.ns-md h3 { color: #e6edf3 !important; font-size: 0.95rem !important; font-weight: 600 !important; }
.ns-md code { background: #21262d !important; color: #79c0ff !important; border-radius: 4px; padding: 2px 6px; }

/* ── Feedback / Reward display ── */
.ns-feedback p, .ns-feedback li, .ns-feedback td, .ns-feedback th {
    color: #c9d1d9 !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', monospace !important;
    line-height: 1.7 !important;
}
.ns-feedback strong { color: #38bdf8 !important; }
.ns-feedback h2, .ns-feedback h3 { color: #f0f6fc !important; }
.ns-feedback code { background: #21262d !important; color: #79c0ff !important; border-radius: 4px; padding: 2px 6px; }
.ns-feedback table { border-collapse: collapse; width: 100%; }
.ns-feedback td, .ns-feedback th { border: 1px solid #30363d !important; padding: 6px 10px !important; }
.ns-feedback th { background: #161b22 !important; color: #8b949e !important; }

/* ── Meals plan display ── */
.meals-md p { color: #c9d1d9 !important; font-size: 0.875rem !important; line-height: 1.8 !important; }
.meals-md em { color: #8b949e !important; }

/* ── Status bar ── */
.meal-status p, .meal-status h3 {
    color: #e6edf3 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px !important;
}

/* ── Allergy section ── */
.ns-allergy {
    background: rgba(220, 38, 38, 0.06) !important;
    border: 1px solid rgba(220, 38, 38, 0.2) !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
.ns-allergy label span { color: #fca5a5 !important; font-size: 0.78rem !important; }

/* ── Ingredient slot cards ── */
.ing-slot {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
.ing-slot label span { color: #6e7681 !important; font-size: 0.75rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Submit button ── */
.btn-submit {
    background: linear-gradient(135deg, #0369a1, #0891b2) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    padding: 14px !important;
    box-shadow: 0 4px 14px rgba(3, 105, 161, 0.4) !important;
    transition: all 0.2s ease !important;
}
.btn-submit:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(3, 105, 161, 0.5) !important;
}

/* ── Reset button ── */
.btn-reset {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    transition: all 0.2s ease !important;
}
.btn-reset:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
}

/* ── Divider ── */
hr { border-color: #21262d !important; margin: 16px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #38bdf8; }

/* ── Checkbox group ── */
.form { background: transparent !important; }
input[type="checkbox"] { accent-color: #38bdf8 !important; }
"""

# ─── Logic ────────────────────────────────────────────────────────────────────

def reset_env(difficulty, custom_budget, allergies_selected):
    global current_env
    task_config = TASKS.get(difficulty.lower(), TASKS["easy"]).copy()
    task_config.pop("grader", None)

    if custom_budget and custom_budget > 0:
        task_config["budget"] = custom_budget

    task_allergies = task_config.get("allergies", []) or []
    ui_allergies   = [a for a in (allergies_selected or []) if a]
    merged         = list(set(task_allergies + ui_allergies))
    task_config["allergies"] = merged

    current_env = NutrisyncEnv(**task_config, seed=42)
    obs   = current_env.reset()
    state = current_env.get_state()

    allergy_str = ", ".join(merged) if merged else "None"
    info_msg = (
        f"Environment ready  —  Task: **{difficulty.upper()}**  |  "
        f"Budget: Rs.{task_config['budget']}  |  "
        f"Calories: {task_config['calorie_target']} kcal  |  "
        f"Blocked: `{allergy_str}`"
    )
    return format_ui(obs, state) + (info_msg,)


def step_env(*args):
    global current_env
    if not current_env or current_env.get_state().done:
        return ("—", "—", "—", "_No meals yet._", "Reset the environment first.", "Reset the environment before submitting meals.")

    items = []
    for i in range(0, len(args), 3):
        ing    = args[i]
        qty    = args[i + 1]
        method = args[i + 2]
        if ing and qty and qty > 0:
            method = method if method in VALID_METHODS else "boiled"
            items.append(IngredientItem(ingredient=ing, quantity=float(qty), cooking_method=method))

    action = NutrisyncAction(items=items)
    obs, reward, done, info = current_env.step(action)
    state = current_env.get_state()

    ui_out = format_ui(obs, state)

    gate_note = ""
    if state.budget_gate <= 0:
        gate_note = "  **[BUDGET BLOWN — episode score = 0]**"
    elif state.budget_gate < 1.0:
        gate_note = "  [Budget gate warning: 0–3% over]"

    fail_note = "  **[HARD FAIL: diet / allergy / quota violated]**" if state.hard_fail else ""
    satiety_note = "in range" if 40 <= obs.satiety <= 80 else "OUT OF RANGE"

    clusters = ", ".join(f"{c} x{n}" for c, n in state.cluster_usage_counts.items())

    feedback = (
        f"**Step reward:** `{reward.score:.4f}`  |  "
        f"**Satiety:** `{obs.satiety:.1f}/100` ({satiety_note})  |  "
        f"**Budget left:** `Rs.{obs.budget_left:.2f}`"
        f"{gate_note}{fail_note}\n\n"
        f"**Clusters used:** {clusters or '—'}\n\n"
        f"{reward.feedback}"
    )

    if done:
        grader      = TASKS[state.constraints.difficulty]["grader"]
        final_grade = grader(state)
        pct         = final_grade * 100
        bar_n       = int(round(pct / 5))
        bar         = "█" * bar_n + "░" * (20 - bar_n)
        feedback = (
            f"## Episode Complete\n\n"
            f"**Final Score:** `{final_grade:.4f}` / 1.0\n\n"
            f"`[{bar}]` **{pct:.1f}%**\n\n"
            f"| Metric | Actual | Target |\n"
            f"|---|---|---|\n"
            f"| Calories | {state.total_calories_consumed:.0f} kcal | {state.constraints.calorie_target:.0f} kcal |\n"
            f"| Protein | {state.total_protein_consumed:.1f} g | {state.constraints.protein_target or 0:.1f} g |\n"
            f"| Budget spent | Rs.{state.total_cost_spent:.2f} | Rs.{state.constraints.budget:.2f} |\n"
            f"| Hard fail | {'Yes' if state.hard_fail else 'No'} | — |\n\n"
            f"_Reset to start a new episode._"
        )

    return ui_out + (feedback,)


def format_ui(obs, state):
    calories_txt = f"{state.total_calories_consumed:.0f} / {state.constraints.calorie_target:.0f} kcal"
    protein_txt  = f"{state.total_protein_consumed:.1f} / {state.constraints.protein_target or 0:.1f} g"
    budget_txt   = f"Rs.{state.total_cost_spent:.2f} / Rs.{state.constraints.budget:.2f}"

    lines = []
    for m, its in state.meals_built.items():
        if its:
            parts = ",  ".join(f"**{it.ingredient}** {it.quantity:.0f}g ({it.cooking_method})" for it in its)
            lines.append(f"**{m.capitalize()}**  —  {parts}")

    meals_text  = "\n\n".join(lines) if lines else "_No meals planned yet._"
    meal_status = (
        f"### Now planning:  {obs.current_meal.upper()}"
        if not obs.done else "### All meals planned."
    )
    return calories_txt, protein_txt, budget_txt, meals_text, meal_status


# ─── Build UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="NutriSync V2") as ui:

    gr.HTML(f"<style>{CSS}</style>")

    # Header
    gr.HTML("""
        <div class="ns-header">
            <h1>NutriSync V2</h1>
            <p>
                OpenEnv RL Environment &mdash; Plan 4 meals (Breakfast &rarr; Lunch &rarr; Dinner &rarr; Snack)
                satisfying calorie, protein, and budget targets. &nbsp;Scores are in <strong style="color:#38bdf8">[0, 1]</strong>.
            </p>
        </div>
    """)

    # Setup
    with gr.Group(elem_classes=["ns-card"]):
        gr.HTML('<p class="ns-label">Environment Setup</p>')
        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard"], value="easy",
                label="Difficulty",
                info="easy = omnivore  Rs.250 | medium = vegetarian  Rs.200 | hard = vegan  Rs.160"
            )
            custom_budget = gr.Number(
                label="Budget Override (Rs.)  —  leave 0 for task default",
                value=0, precision=0, minimum=0
            )
            reset_btn = gr.Button("Initialise / Reset", elem_classes=["btn-reset"])

        with gr.Group(elem_classes=["ns-allergy"]):
            allergies_input = gr.CheckboxGroup(
                choices=ALL_INGREDIENTS,
                value=[],
                label="Allergy / Restriction Overrides — select ingredients to block",
                info="Any blocked ingredient used in a meal triggers HARD FAIL (score = 0).",
            )

    # Progress + Meal Plan
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["ns-card"]):
            gr.HTML('<p class="ns-label">Progress</p>')
            cal_disp  = gr.Textbox(label="Calories",      interactive=False, elem_classes=["prog-txt"])
            prot_disp = gr.Textbox(label="Protein",       interactive=False, elem_classes=["prog-txt"])
            budg_disp = gr.Textbox(label="Budget Spent",  interactive=False, elem_classes=["prog-txt"])

        with gr.Column(scale=2, elem_classes=["ns-card"]):
            gr.HTML('<p class="ns-label">Today\'s Meal Plan</p>')
            meals_disp = gr.Markdown("_No meals planned yet._", elem_classes=["meals-md"])

    meal_status     = gr.Markdown("### Now planning:  BREAKFAST", elem_classes=["meal-status"])
    reward_feedback = gr.Markdown("_Submit a meal to see reward signals._", elem_classes=["ns-feedback"])

    # Meal Builder
    gr.HTML("<hr/>")
    gr.HTML('<p class="ns-label">Build Your Meal — pick up to 4 ingredients</p>')
    gr.Markdown(
        "Each meal must contain a **protein source**, a **carb source**, and a **fat source** "
        "to earn the meal completeness bonus.",
        elem_classes=["ns-md"]
    )

    meal_inputs = []
    with gr.Row():
        for i in range(4):
            with gr.Column(elem_classes=["ing-slot"]):
                gr.Markdown(f"Slot {i + 1}", elem_classes=["ns-label"])
                ing    = gr.Dropdown(choices=ALL_INGREDIENTS, label="Ingredient", value=None)
                qty    = gr.Number(value=0, label="Quantity (g)", minimum=0, maximum=600)
                method = gr.Dropdown(choices=VALID_METHODS, value="boiled", label="Cooking Method")
                meal_inputs.extend([ing, qty, method])

    submit_btn = gr.Button("Submit Meal", variant="primary", elem_classes=["btn-submit"], size="lg")

    # Wiring
    outputs = [cal_disp, prot_disp, budg_disp, meals_disp, meal_status, reward_feedback]
    reset_btn.click(fn=reset_env, inputs=[difficulty, custom_budget, allergies_input], outputs=outputs)
    submit_btn.click(fn=step_env, inputs=meal_inputs, outputs=outputs)
    ui.load(fn=reset_env, inputs=[difficulty, custom_budget, allergies_input], outputs=outputs)


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
