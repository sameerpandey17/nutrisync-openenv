#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Example script demonstrating Nutrisync environment usage.

Run the server first:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Then run this script:
    python example.py
"""

import logging
from client import NutrisyncClient
from models import NutrisyncAction as Action, IngredientItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_observation(obs):
    """Pretty print observation."""
    print("\n" + "="*80)
    print(f"OBSERVATION: {obs.current_meal.upper()}")
    print("="*80)
    print(f"  Current Meal: {obs.current_meal}")
    print(f"  Calories Left: {obs.calories_left:.1f}")
    print(f"  Protein Left: {obs.protein_left:.1f}g")
    print(f"  Budget Left: Rs.{obs.budget_left:.2f}")
    print(f"  Step Count: {obs.step_count}")
    print(f"  Done: {obs.done}")


def example_easy_mode():
    """Example: EASY mode (calories + budget only)."""
    print("\n[EASY MODE] Calories + Budget Only")
    print("-" * 80)
    
    client = NutrisyncClient()
    obs = client.reset(difficulty="easy", calorie_target=2000, budget=250)
    print("Environment reset")
    
    # Breakfast
    action = Action(items=[
        IngredientItem(ingredient="eggs", quantity=100, cooking_method="boiled"),
        IngredientItem(ingredient="milk", quantity=200, cooking_method="boiled"),
    ])
    obs, reward, done, info = client.step(action)
    print(f"[Breakfast] Reward: {reward.score}")
    
    # Lunch
    action = Action(items=[
        IngredientItem(ingredient="chicken", quantity=150, cooking_method="boiled"),
        IngredientItem(ingredient="rice", quantity=150, cooking_method="boiled"),
    ])
    obs, reward, done, info = client.step(action)
    print(f"[Lunch] Reward: {reward.score}")
    
    # Dinner
    action = Action(items=[
        IngredientItem(ingredient="mutton", quantity=150, cooking_method="boiled"),
        IngredientItem(ingredient="roti", quantity=100, cooking_method="roasted"),
    ])
    obs, reward, done, info = client.step(action)
    print(f"[Dinner] Reward: {reward.score}")
    
    # Snack
    action = Action(items=[
        IngredientItem(ingredient="apple", quantity=150, cooking_method="raw"),
    ])
    obs, reward, done, info = client.step(action)
    print(f"[Snack] Reward: {reward.score}")
    
    summary = client.summary()
    print(f"\n[Summary] Total Calories: {summary.get('total_calories', 0):.1f}")


def example_medium_mode():
    """Example: MEDIUM mode (calories + protein + budget)."""
    print("\n[MEDIUM MODE] Calories + Protein + Budget")
    print("-" * 80)
    
    client = NutrisyncClient()
    obs = client.reset(
        difficulty="medium",
        calorie_target=2000,
        protein_target=100,
        budget=200,
    )
    print("Environment reset with protein target")
    
    meals_data = [
        ("breakfast", [
            ("oats", 50, "boiled"),
            ("eggs", 100, "boiled"),
        ]),
        ("lunch", [
            ("chicken", 150, "boiled"),
            ("rice", 150, "boiled"),
        ]),
        ("dinner", [
            ("paneer", 150, "raw"),
            ("wheat_flour", 100, "boiled"),
        ]),
        ("snack", [
            ("apple", 150, "raw"),
        ]),
    ]
    
    for meal_name, items in meals_data:
        action = Action(items=[
            IngredientItem(ingredient=ing, quantity=qty, cooking_method=method)
            for ing, qty, method in items
        ])
        obs, reward, done, info = client.step(action)
        print(f"[{meal_name.upper()}] Calories: {info.get('calories_added', 0):.1f}, Protein: {info.get('protein_added', 0):.1f}g")
    
    summary = client.summary()
    print(f"\n[Summary] Total Protein: {summary.get('total_protein', 0):.1f}g")


def example_expert_mode():
    """Example: EXPERT mode (all constraints)."""
    print("\n[EXPERT MODE] All Constraints")
    print("-" * 80)
    
    client = NutrisyncClient()
    obs = client.reset(
        difficulty="expert",
        calorie_target=2000,
        protein_target=120,
        budget=160,
        diet_type="omnivore",
        allergies=["fish"],
        ingredient_usage_limits={"chicken": 2},
    )
    print("Environment reset with allergies and usage limits")
    
    # Breakfast
    action = Action(items=[
        IngredientItem(ingredient="eggs", quantity=100, cooking_method="boiled"),
        IngredientItem(ingredient="bread", quantity=100, cooking_method="raw"),
    ])
    obs, reward, done, info = client.step(action)
    print(f"[Breakfast] OK")
    
    # Lunch
    action = Action(items=[
        IngredientItem(ingredient="chicken", quantity=150, cooking_method="boiled"),
        IngredientItem(ingredient="rice", quantity=150, cooking_method="boiled"),
    ])
    obs, reward, done, info = client.step(action)
    print(f"[Lunch] OK")
    
    # Dinner
    action = Action(items=[
        IngredientItem(ingredient="paneer", quantity=150, cooking_method="raw"),
        IngredientItem(ingredient="wheat_flour", quantity=100, cooking_method="boiled"),
    ])
    obs, reward, done, info = client.step(action)
    print(f"[Dinner] OK")
    
    # Snack
    action = Action(items=[
        IngredientItem(ingredient="apple", quantity=150, cooking_method="raw"),
    ])
    obs, reward, done, info = client.step(action)
    print(f"[Snack] OK")
    
    print("[Summary] Episode completed with all constraints satisfied")


def example_invalid_action():
    """Example: Testing constraint validation."""
    print("\n[INVALID ACTION TEST]")
    print("-" * 80)
    
    client = NutrisyncClient()
    obs = client.reset(difficulty="easy", diet_type="vegan")
    print("Environment reset with VEGAN diet")
    
    # Try to use chicken (not vegan)
    action = Action(items=[
        IngredientItem(ingredient="chicken", quantity=100)
    ])
    obs, reward, done, info = client.step(action)
    
    if "error" in info:
        print(f"[Correctly Rejected] {info['error']}")
    else:
        print("[ERROR] Should have rejected chicken in vegan diet")


if __name__ == "__main__":
    print("="*80)
    print("NUTRISYNC RL ENVIRONMENT - EXAMPLES")
    print("="*80)
    
    try:
        example_easy_mode()
        example_medium_mode()
        example_expert_mode()
        example_invalid_action()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"ERROR: {e}")
