import sys
import os
import gradio as gr
import pandas as pd

# Fix path to load env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

from server.environment import NutrisyncEnv
from models import NutrisyncAction, IngredientItem
from server.tasks import TASKS

# Global state to keep the active environment instance
current_env = None

def get_ingredient_choices():
    # Helper to load ingredients from the test database
    temp_env = NutrisyncEnv()
    db = temp_env.nutrition_db
    return sorted(list(db.keys()))

def reset_env(difficulty, custom_budget=0):
    global current_env
    task_config = TASKS.get(difficulty.lower(), TASKS['easy']).copy()
    task_config.pop('grader', None)
    
    if custom_budget and custom_budget > 0:
        task_config['budget'] = custom_budget
    
    current_env = NutrisyncEnv(**task_config, seed=42)
    obs = current_env.reset()
    state = current_env.get_state()
    return format_ui(obs, state)

def step_env(*args):
    global current_env
    if not current_env or current_env.get_state().done:
        return reset_env("easy")  # fallback
    
    # args comes from Gradio UI inputs (e.g., dropdown, number, dropdown, number ...)
    items = []
    # The first 2 inputs are difficulty and custom_budget, skip them
    # Actually, the way Gradio works, we need to be careful with indexing
    # Let's rebuild the input list in the Blocks section
    
    # Wait, in the click() call, we specify inputs. 
    # Let's make sure the inputs match.
    
    # The inputs list in the UI currently has: [dropdown, number, dropdown, number ...] 
    # but reset_env takes [difficulty, custom_budget].
    # I will change step_env to expect only the ingredient items.
    
    items_data = args
    items = []
    for i in range(0, len(items_data), 2):
        ing = items_data[i]
        qty = items_data[i+1]
        if ing and qty and qty > 0:
            items.append(IngredientItem(ingredient=ing, quantity=float(qty)))
    
    action = NutrisyncAction(items=items)
    obs, reward, done, info = current_env.step(action)
    state = current_env.get_state()
    
    ui_out = format_ui(obs, state)
    reward_str = f"Latest Reward: {reward.score:.2f} | Feedback: {reward.feedback}"
    if done:
        grader = TASKS[state.constraints.difficulty]['grader']
        final_grade = grader(state)
        reward_str += f"\n\nEPISODE FINISHED! FINAL GRADE: {final_grade*100:.1f} / 100"
    
    return ui_out + (reward_str,)

def format_ui(obs, state):
    calories_txt = f"{state.total_calories_consumed:.1f} / {state.constraints.calorie_target:.1f} kcal"
    protein_txt = f"{state.total_protein_consumed:.1f} / {state.constraints.protein_target or 0:.1f} g"
    budget_txt = f"{state.total_cost_spent:.2f} / {state.constraints.budget:.2f}"
    
    meals_summary = []
    for m, items in state.meals_built.items():
        if items:
            s = f"**{m.capitalize()}**: " + ", ".join([f"{i.ingredient} ({i.quantity}g)" for i in items])
            meals_summary.append(s)
    
    meals_text = "\n".join(meals_summary) if meals_summary else "No meals planned yet."
    meal_status = f"Now planning: {obs.current_meal.upper()}" if not obs.done else "All Meals Planned!"
    
    return calories_txt, protein_txt, budget_txt, meals_text, meal_status

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald")) as ui:
    gr.Markdown("#  NutriSync Meal Planner Agent Playground")
    gr.Markdown("Agentic interactive interface for the OpenEnv NutriSync environment. Build a 4-meal dietary plan satisfying constraints.")
    
    with gr.Row():
        difficulty = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task Difficulty")
        custom_budget = gr.Number(label="Custom Budget Override ( - 0 for default)", value=0, precision=0)
        reset_btn = gr.Button(" Initialize / Reset Environment")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Current Progress")
            cal_disp = gr.Textbox(label="Calories Consumed")
            prot_disp = gr.Textbox(label="Protein Consumed")
            budg_disp = gr.Textbox(label="Budget Spent")
        with gr.Column():
            gr.Markdown("### Day's Plan")
            meals_disp = gr.Markdown()
    
    gr.Markdown("---")
    meal_status = gr.Markdown("### Now planning: BREAKFAST")
    reward_feedback = gr.Textbox(label="Reward Signal & Feedback", lines=2)
    
    ingredients_list = get_ingredient_choices()
    
    # Input area: up to 5 items per meal
    meal_inputs = []
    with gr.Row():
        for i in range(4):
            with gr.Column():
                ing = gr.Dropdown(choices=ingredients_list, label=f"Item {i+1}")
                qty = gr.Number(value=0, label="Quantity (g)")
                meal_inputs.extend([ing, qty])
                
    submit_btn = gr.Button("Submit Meal", variant="primary")
    
    # Wiring
    reset_outputs = [cal_disp, prot_disp, budg_disp, meals_disp, meal_status]
    reset_btn.click(fn=reset_env, inputs=[difficulty, custom_budget], outputs=reset_outputs)
    
    step_outputs = reset_outputs + [reward_feedback]
    submit_btn.click(fn=step_env, inputs=meal_inputs, outputs=step_outputs)
    
    # Kickoff auto-reset
    ui.load(fn=reset_env, inputs=[difficulty, custom_budget], outputs=reset_outputs)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
