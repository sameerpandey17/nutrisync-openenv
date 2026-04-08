# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Production-grade Nutrisync RL Environment Core Logic  V2.

V2 upgrades:
  - 50-ingredient database extended with fiber, sugars, micronutrients,
    glycemic index, carbs, fat, and semantic cluster tags
  - Hard-fail kill switch (allergy / diet-mode / quota violations)
  - Per-step cumulative budget gate (irreversible at >5% overage)
  - Persistent satiety state with satiety-window penalty
  - Cooking-method nutritional modifiers (METHOD_MODIFIERS)
  - Semantic cluster tracking (cluster_usage_counts)
  - Temporal protein-pace constraints (action-space filtering)
  - Seasonal availability sampling at episode reset
  - Per-step accumulators for calorie pacing, GL penalty,
    meal completeness, and cooking coherence
"""

import logging
import random
import uuid
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

try:
    from models import (
        NutrisyncAction, NutrisyncObservation, Reward, RewardBreakdown,
        NutrisyncState, ConstraintConfig, IngredientItem, NutritionInfo, MealTarget
    )
except ImportError:
    from ..models import (
        NutrisyncAction, NutrisyncObservation, Reward, RewardBreakdown,
        NutrisyncState, ConstraintConfig, IngredientItem, NutritionInfo, MealTarget
    )

try:
    from reward import RewardEngine
except ImportError:
    from .reward import RewardEngine

try:
    from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironment
except ImportError:
    OpenEnvEnvironment = object  # Graceful fallback if openenv not installed

logger = logging.getLogger(__name__)


# ============================================================================
# COOKING METHOD MODIFIERS
# ============================================================================

METHOD_MODIFIERS: Dict[str, Dict[str, float]] = {
    "raw":       {"cal": 1.00, "protein": 1.00, "vitamin_c": 1.00, "fiber": 1.00},
    "boiled":    {"cal": 1.00, "protein": 1.00, "vitamin_c": 0.80, "fiber": 0.90},
    "steamed":   {"cal": 1.00, "protein": 1.00, "vitamin_c": 0.90, "fiber": 1.00},
    "sauteed":   {"cal": 1.15, "protein": 1.00, "vitamin_c": 0.85, "fiber": 0.95},
    "fried":     {"cal": 1.45, "protein": 0.95, "vitamin_c": 0.60, "fiber": 0.80},
    "roasted":   {"cal": 1.05, "protein": 1.00, "vitamin_c": 0.70, "fiber": 1.00},
    "fermented": {"cal": 0.95, "protein": 1.05, "vitamin_c": 1.10, "fiber": 1.10},
}

# Per-ingredient optimal cooking method for coherence bonus
OPTIMAL_METHODS: Dict[str, str] = {
    # Proteins
    "chicken":      "boiled",
    "mutton":       "boiled",
    "fish_rohu":    "steamed",
    "eggs":         "boiled",
    "paneer":       "raw",
    "tofu":         "steamed",
    # Legumes
    "moong_dal":    "boiled",
    "toor_dal":     "boiled",
    "chana_dal":    "boiled",
    "masoor_dal":   "boiled",
    "rajma":        "boiled",
    "chickpeas":    "boiled",
    # Grains
    "rice":         "boiled",
    "wheat_flour":  "boiled",
    "roti":         "roasted",
    "oats":         "boiled",
    "poha":         "sauteed",
    "semolina":     "boiled",
    "bread":        "raw",
    "millet_bajra": "roasted",
    "ragi":         "roasted",
    # Vegetables
    "potato":       "boiled",
    "onion":        "sauteed",
    "tomato":       "sauteed",
    "spinach":      "steamed",
    "cauliflower":  "steamed",
    "cabbage":      "steamed",
    "brinjal":      "roasted",
    "lady_finger":  "sauteed",
    "bottle_gourd": "boiled",
    "carrot":       "raw",
    "green_peas":   "steamed",
    "capsicum":     "raw",
    # Fruits
    "banana":       "raw",
    "apple":        "raw",
    "mango":        "raw",
    "papaya":       "raw",
    "guava":        "raw",
    "coconut":      "raw",
    # Dairy
    "milk":         "boiled",
    "curd":         "raw",
    "ghee":         "raw",
    "butter":       "raw",
    "cheese":       "raw",
    # Fats & condiments
    "mustard_oil":  "raw",
    "coconut_oil":  "raw",
    "groundnut_oil":"raw",
    "sugar":        "raw",
    "jaggery":      "raw",
    "honey":        "raw",
}

# Meal calorie distribution targets (V2 spec)
MEAL_CALORIE_FRACTIONS: Dict[str, float] = {
    "breakfast": 0.25,
    "lunch":     0.35,
    "dinner":    0.30,
    "snack":     0.10,
}

# Protein pace targets (cumulative fraction by end of each meal)
PROTEIN_PACE_TARGETS: Dict[str, float] = {
    "breakfast": 0.25,
    "lunch":     0.60,
    "dinner":    0.90,   # (spec says 0.70 for snack, 1.0 for dinner; dinner comes before snack)
    "snack":     1.00,
}

MIN_PROTEIN_PER_ING: float = 15.0        # g  used for temporal action filtering
SATIETY_LOW: float = 40.0
SATIETY_HIGH: float = 80.0
SATIETY_PENALTY: float = -0.3

# ============================================================================
# INDIAN NUTRITION DATABASE  V2 (50 ingredients, per 100g cooked)
# ============================================================================

NUTRITION_DATABASE: Dict[str, NutritionInfo] = {
    # ==================== PROTEINS (7 items) ====================
    "chicken": NutritionInfo(
        calories=239, protein=27.0, cost=40,
        carb_g=0.0,  fat_g=13.6,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=1.1, calcium_mg=11,  vitamin_c_mg=0.0,
        glycemic_index=0, cluster="proteins"
    ),
    "mutton": NutritionInfo(
        calories=294, protein=25.0, cost=80,
        carb_g=0.0,  fat_g=21.0,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=2.7, calcium_mg=14,  vitamin_c_mg=0.0,
        glycemic_index=0, cluster="proteins"
    ),
    "fish_rohu": NutritionInfo(
        calories=97,  protein=17.0, cost=30,
        carb_g=0.0,  fat_g=2.7,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=1.0, calcium_mg=65,  vitamin_c_mg=0.0,
        glycemic_index=0, cluster="proteins"
    ),
    "eggs": NutritionInfo(
        calories=155, protein=13.0, cost=15,
        carb_g=1.1,  fat_g=11.0,
        fiber_g=0.0, simple_sugar_g=1.1,
        iron_mg=1.8, calcium_mg=56,  vitamin_c_mg=0.0,
        glycemic_index=0, cluster="proteins"
    ),
    "tofu": NutritionInfo(
        calories=76,  protein=8.0,  cost=25,
        carb_g=1.9,  fat_g=4.8,
        fiber_g=0.3, simple_sugar_g=0.7,
        iron_mg=1.8, calcium_mg=350, vitamin_c_mg=0.1,
        glycemic_index=15, cluster="proteins"
    ),

    # ==================== LEGUMES (7 items) ====================
    "moong_dal": NutritionInfo(
        calories=347, protein=24.0, cost=12,
        carb_g=60.0, fat_g=1.2,
        fiber_g=7.6, simple_sugar_g=1.2,
        iron_mg=4.4, calcium_mg=73,  vitamin_c_mg=1.0,
        glycemic_index=25, cluster="legumes"
    ),
    "toor_dal": NutritionInfo(
        calories=343, protein=22.0, cost=14,
        carb_g=60.0, fat_g=1.5,
        fiber_g=5.1, simple_sugar_g=1.2,
        iron_mg=5.3, calcium_mg=54,  vitamin_c_mg=1.9,
        glycemic_index=29, cluster="legumes"
    ),
    "chana_dal": NutritionInfo(
        calories=360, protein=20.0, cost=13,
        carb_g=60.0, fat_g=5.0,
        fiber_g=7.2, simple_sugar_g=2.0,
        iron_mg=4.0, calcium_mg=69,  vitamin_c_mg=1.5,
        glycemic_index=28, cluster="legumes"
    ),
    "masoor_dal": NutritionInfo(
        calories=352, protein=25.0, cost=11,
        carb_g=59.0, fat_g=1.1,
        fiber_g=7.9, simple_sugar_g=1.8,
        iron_mg=7.6, calcium_mg=65,  vitamin_c_mg=1.5,
        glycemic_index=26, cluster="legumes"
    ),
    "rajma": NutritionInfo(
        calories=333, protein=24.0, cost=16,
        carb_g=57.0, fat_g=1.5,
        fiber_g=6.4, simple_sugar_g=1.0,
        iron_mg=2.9, calcium_mg=143, vitamin_c_mg=1.0,
        glycemic_index=24, cluster="legumes"
    ),
    "chickpeas": NutritionInfo(
        calories=164, protein=9.0,  cost=14,
        carb_g=27.0, fat_g=2.6,
        fiber_g=7.6, simple_sugar_g=0.0,
        iron_mg=3.0, calcium_mg=105, vitamin_c_mg=1.3,
        glycemic_index=28, cluster="legumes"
    ),
    "paneer": NutritionInfo(
        calories=265, protein=18.0, cost=50,
        carb_g=1.2,  fat_g=20.8,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=0.2, calcium_mg=480, vitamin_c_mg=0.0,
        glycemic_index=0, cluster="dairy"
    ),

    # ==================== GRAINS  WHOLE (5 items) ====================
    "wheat_flour": NutritionInfo(
        calories=340, protein=12.0, cost=5,
        carb_g=69.0, fat_g=1.5,
        fiber_g=2.7, simple_sugar_g=0.5,
        iron_mg=3.5, calcium_mg=23,  vitamin_c_mg=0.0,
        glycemic_index=50, cluster="whole_grains"
    ),
    "roti": NutritionInfo(
        calories=297, protein=8.0,  cost=4,
        carb_g=57.0, fat_g=3.7,
        fiber_g=2.4, simple_sugar_g=0.3,
        iron_mg=2.7, calcium_mg=30,  vitamin_c_mg=0.0,
        glycemic_index=52, cluster="whole_grains"
    ),
    "oats": NutritionInfo(
        calories=389, protein=17.0, cost=18,
        carb_g=66.0, fat_g=7.0,
        fiber_g=10.6, simple_sugar_g=0.8,
        iron_mg=3.6, calcium_mg=54,  vitamin_c_mg=0.0,
        glycemic_index=55, cluster="whole_grains"
    ),
    "millet_bajra": NutritionInfo(
        calories=378, protein=11.0, cost=6,
        carb_g=72.0, fat_g=5.0,
        fiber_g=1.3, simple_sugar_g=0.5,
        iron_mg=8.0, calcium_mg=42,  vitamin_c_mg=0.0,
        glycemic_index=50, cluster="whole_grains"
    ),
    "ragi": NutritionInfo(
        calories=328, protein=7.0,  cost=8,
        carb_g=72.0, fat_g=1.5,
        fiber_g=3.6, simple_sugar_g=0.5,
        iron_mg=3.9, calcium_mg=344, vitamin_c_mg=0.0,
        glycemic_index=68, cluster="whole_grains"
    ),

    # ==================== GRAINS  REFINED (4 items) ====================
    "rice": NutritionInfo(
        calories=130, protein=2.7,  cost=8,
        carb_g=28.0, fat_g=0.3,
        fiber_g=0.4, simple_sugar_g=0.1,
        iron_mg=0.4, calcium_mg=10,  vitamin_c_mg=0.0,
        glycemic_index=72, cluster="refined_carbs"
    ),
    "poha": NutritionInfo(
        calories=346, protein=6.6,  cost=8,
        carb_g=77.0, fat_g=0.7,
        fiber_g=0.9, simple_sugar_g=0.5,
        iron_mg=2.0, calcium_mg=14,  vitamin_c_mg=0.0,
        glycemic_index=65, cluster="refined_carbs"
    ),
    "semolina": NutritionInfo(
        calories=360, protein=13.0, cost=7,
        carb_g=73.0, fat_g=1.0,
        fiber_g=1.2, simple_sugar_g=0.5,
        iron_mg=1.5, calcium_mg=17,  vitamin_c_mg=0.0,
        glycemic_index=60, cluster="refined_carbs"
    ),
    "bread": NutritionInfo(
        calories=265, protein=9.0,  cost=8,
        carb_g=49.0, fat_g=3.5,
        fiber_g=2.7, simple_sugar_g=5.0,
        iron_mg=2.0, calcium_mg=60,  vitamin_c_mg=0.0,
        glycemic_index=71, cluster="refined_carbs"
    ),

    # ==================== VEGETABLES (12 items) ====================
    "potato": NutritionInfo(
        calories=77,  protein=2.0,  cost=3,
        carb_g=17.0, fat_g=0.1,
        fiber_g=1.8, simple_sugar_g=0.9,
        iron_mg=0.6, calcium_mg=10,  vitamin_c_mg=12.0,
        glycemic_index=78, cluster="vegetables"
    ),
    "onion": NutritionInfo(
        calories=40,  protein=1.1,  cost=4,
        carb_g=9.0,  fat_g=0.1,
        fiber_g=1.7, simple_sugar_g=4.2,
        iron_mg=0.2, calcium_mg=23,  vitamin_c_mg=7.4,
        glycemic_index=35, cluster="vegetables"
    ),
    "tomato": NutritionInfo(
        calories=18,  protein=0.9,  cost=4,
        carb_g=3.9,  fat_g=0.2,
        fiber_g=1.2, simple_sugar_g=2.6,
        iron_mg=0.4, calcium_mg=10,  vitamin_c_mg=27.0,
        glycemic_index=30, cluster="vegetables"
    ),
    "spinach": NutritionInfo(
        calories=23,  protein=2.9,  cost=6,
        carb_g=3.6,  fat_g=0.4,
        fiber_g=2.2, simple_sugar_g=0.4,
        iron_mg=2.7, calcium_mg=99,  vitamin_c_mg=28.0,
        glycemic_index=15, cluster="leafy_greens"
    ),
    "cauliflower": NutritionInfo(
        calories=25,  protein=1.9,  cost=5,
        carb_g=5.0,  fat_g=0.3,
        fiber_g=2.0, simple_sugar_g=1.9,
        iron_mg=0.4, calcium_mg=22,  vitamin_c_mg=48.0,
        glycemic_index=15, cluster="vegetables"
    ),
    "cabbage": NutritionInfo(
        calories=25,  protein=1.3,  cost=4,
        carb_g=5.8,  fat_g=0.1,
        fiber_g=2.5, simple_sugar_g=3.5,
        iron_mg=0.5, calcium_mg=40,  vitamin_c_mg=36.0,
        glycemic_index=15, cluster="vegetables"
    ),
    "brinjal": NutritionInfo(
        calories=25,  protein=1.0,  cost=5,
        carb_g=5.9,  fat_g=0.2,
        fiber_g=3.0, simple_sugar_g=3.5,
        iron_mg=0.2, calcium_mg=15,  vitamin_c_mg=5.0,
        glycemic_index=15, cluster="vegetables"
    ),
    "lady_finger": NutritionInfo(
        calories=33,  protein=1.9,  cost=6,
        carb_g=7.5,  fat_g=0.2,
        fiber_g=3.2, simple_sugar_g=1.5,
        iron_mg=0.4, calcium_mg=82,  vitamin_c_mg=24.0,
        glycemic_index=20, cluster="vegetables"
    ),
    "bottle_gourd": NutritionInfo(
        calories=15,  protein=0.6,  cost=4,
        carb_g=3.4,  fat_g=0.0,
        fiber_g=0.5, simple_sugar_g=2.0,
        iron_mg=0.3, calcium_mg=27,  vitamin_c_mg=10.0,
        glycemic_index=15, cluster="vegetables"
    ),
    "carrot": NutritionInfo(
        calories=41,  protein=0.9,  cost=5,
        carb_g=10.0, fat_g=0.2,
        fiber_g=2.8, simple_sugar_g=4.7,
        iron_mg=0.3, calcium_mg=33,  vitamin_c_mg=6.0,
        glycemic_index=35, cluster="vegetables"
    ),
    "green_peas": NutritionInfo(
        calories=81,  protein=5.4,  cost=8,
        carb_g=14.0, fat_g=0.4,
        fiber_g=5.1, simple_sugar_g=4.0,
        iron_mg=1.5, calcium_mg=25,  vitamin_c_mg=40.0,
        glycemic_index=48, cluster="vegetables"
    ),
    "capsicum": NutritionInfo(
        calories=31,  protein=1.0,  cost=8,
        carb_g=6.0,  fat_g=0.3,
        fiber_g=2.3, simple_sugar_g=2.6,
        iron_mg=0.4, calcium_mg=7,   vitamin_c_mg=127.0,
        glycemic_index=15, cluster="vegetables"
    ),

    # ==================== FRUITS (6 items) ====================
    "banana": NutritionInfo(
        calories=89,  protein=1.1,  cost=3,
        carb_g=23.0, fat_g=0.3,
        fiber_g=2.6, simple_sugar_g=12.0,
        iron_mg=0.3, calcium_mg=5,   vitamin_c_mg=8.7,
        glycemic_index=52, cluster="fruits"
    ),
    "apple": NutritionInfo(
        calories=52,  protein=0.3,  cost=12,
        carb_g=14.0, fat_g=0.2,
        fiber_g=2.4, simple_sugar_g=10.0,
        iron_mg=0.1, calcium_mg=6,   vitamin_c_mg=5.0,
        glycemic_index=36, cluster="fruits"
    ),
    "mango": NutritionInfo(
        calories=60,  protein=0.8,  cost=10,
        carb_g=15.0, fat_g=0.4,
        fiber_g=1.6, simple_sugar_g=14.0,
        iron_mg=0.1, calcium_mg=11,  vitamin_c_mg=36.0,
        glycemic_index=56, cluster="fruits"
    ),
    "papaya": NutritionInfo(
        calories=43,  protein=0.5,  cost=5,
        carb_g=11.0, fat_g=0.3,
        fiber_g=1.7, simple_sugar_g=5.9,
        iron_mg=0.3, calcium_mg=20,  vitamin_c_mg=61.8,
        glycemic_index=60, cluster="fruits"
    ),
    "guava": NutritionInfo(
        calories=68,  protein=2.6,  cost=6,
        carb_g=14.0, fat_g=1.0,
        fiber_g=5.4, simple_sugar_g=5.0,
        iron_mg=0.3, calcium_mg=18,  vitamin_c_mg=228.0,
        glycemic_index=12, cluster="fruits"
    ),
    "coconut": NutritionInfo(
        calories=354, protein=3.3,  cost=10,
        carb_g=15.0, fat_g=33.5,
        fiber_g=9.0, simple_sugar_g=6.0,
        iron_mg=2.4, calcium_mg=14,  vitamin_c_mg=0.0,
        glycemic_index=45, cluster="vegetables"
    ),

    # ==================== DAIRY (5 items) ====================
    "milk": NutritionInfo(
        calories=61,  protein=3.2,  cost=6,
        carb_g=4.8,  fat_g=3.3,
        fiber_g=0.0, simple_sugar_g=4.8,
        iron_mg=0.1, calcium_mg=120, vitamin_c_mg=1.0,
        glycemic_index=30, cluster="dairy"
    ),
    "curd": NutritionInfo(
        calories=60,  protein=3.5,  cost=8,
        carb_g=3.4,  fat_g=3.3,
        fiber_g=0.0, simple_sugar_g=4.0,
        iron_mg=0.1, calcium_mg=150, vitamin_c_mg=0.0,
        glycemic_index=36, cluster="dairy"
    ),
    "ghee": NutritionInfo(
        calories=900, protein=0.0,  cost=80,
        carb_g=0.0,  fat_g=99.8,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=0.0, calcium_mg=1,   vitamin_c_mg=0.0,
        glycemic_index=0, cluster="fats"
    ),
    "butter": NutritionInfo(
        calories=717, protein=0.5,  cost=70,
        carb_g=0.1,  fat_g=81.0,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=0.0, calcium_mg=11,  vitamin_c_mg=0.0,
        glycemic_index=0, cluster="fats"
    ),
    "cheese": NutritionInfo(
        calories=349, protein=25.0, cost=60,
        carb_g=1.3,  fat_g=27.0,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=0.2, calcium_mg=721, vitamin_c_mg=0.0,
        glycemic_index=0, cluster="dairy"
    ),

    # ==================== FATS & CONDIMENTS (6 items) ====================
    "mustard_oil": NutritionInfo(
        calories=884, protein=0.0,  cost=18,
        carb_g=0.0,  fat_g=100.0,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=0.0, calcium_mg=0,   vitamin_c_mg=0.0,
        glycemic_index=0, cluster="fats"
    ),
    "coconut_oil": NutritionInfo(
        calories=862, protein=0.0,  cost=22,
        carb_g=0.0,  fat_g=100.0,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=0.0, calcium_mg=0,   vitamin_c_mg=0.0,
        glycemic_index=0, cluster="fats"
    ),
    "groundnut_oil": NutritionInfo(
        calories=884, protein=0.0,  cost=16,
        carb_g=0.0,  fat_g=100.0,
        fiber_g=0.0, simple_sugar_g=0.0,
        iron_mg=0.0, calcium_mg=0,   vitamin_c_mg=0.0,
        glycemic_index=0, cluster="fats"
    ),
    "sugar": NutritionInfo(
        calories=387, protein=0.0,  cost=5,
        carb_g=100.0, fat_g=0.0,
        fiber_g=0.0, simple_sugar_g=99.0,
        iron_mg=0.0, calcium_mg=1,   vitamin_c_mg=0.0,
        glycemic_index=65, cluster="simple_sugars"
    ),
    "jaggery": NutritionInfo(
        calories=383, protein=0.4,  cost=8,
        carb_g=98.0, fat_g=0.1,
        fiber_g=0.0, simple_sugar_g=65.0,
        iron_mg=11.4, calcium_mg=80,  vitamin_c_mg=0.0,
        glycemic_index=84, cluster="simple_sugars"
    ),
    "honey": NutritionInfo(
        calories=304, protein=0.3,  cost=30,
        carb_g=82.0, fat_g=0.0,
        fiber_g=0.2, simple_sugar_g=82.0,
        iron_mg=0.4, calcium_mg=6,   vitamin_c_mg=0.5,
        glycemic_index=58, cluster="simple_sugars"
    ),
}

# Ingredient IDs for seasonal sampling
ALL_INGREDIENT_IDS: List[str] = list(NUTRITION_DATABASE.keys())

# Diet exclusions
_NON_VEG = {"chicken", "mutton", "fish_rohu"}
_NON_VEGAN = {
    "chicken", "mutton", "fish_rohu", "eggs", "paneer",
    "milk", "curd", "ghee", "butter", "cheese", "honey",
}

DIET_COMPATIBILITY: Dict[str, List[str]] = {
    "omnivore":   list(NUTRITION_DATABASE.keys()),
    "vegetarian": [k for k in NUTRITION_DATABASE if k not in _NON_VEG],
    "vegan":      [k for k in NUTRITION_DATABASE if k not in _NON_VEGAN],
}

MEAL_ORDER: List[Literal["breakfast", "lunch", "dinner", "snack"]] = [
    "breakfast", "lunch", "dinner", "snack"
]

# Neutral filler substituted for unavailable ingredient
UNAVAILABLE_FILLER = "rice"


# ============================================================================
# NUTRISYNC ENVIRONMENT  V2
# ============================================================================

class NutrisyncEnv(OpenEnvEnvironment):
    """
    Production-grade Nutrisync RL Environment  V2.

    Sequential meal planning with 4 steps (breakfast  lunch  dinner  snack).
    Implements all V2 mechanics:
      - Multiplicative kill switches (hard_fail, budget_gate)
      - Persistent satiety state
      - Semantic cluster spam detection
      - Temporal protein-pace constraints
      - Seasonal ingredient availability
      - Cooking-method nutritional modifiers
    """

    def __init__(
        self,
        difficulty: Literal["easy", "medium", "expert"] = "medium",
        calorie_target: float = 2000.0,
        protein_target: Optional[float] = None,
        budget: float = 200.0,
        diet_type: Literal["omnivore", "vegetarian", "vegan"] = "omnivore",
        allergies: Optional[List[str]] = None,
        seed: Optional[int] = None,
        ingredient_usage_limits: Optional[Dict[str, int]] = None,
        # V2 feature flags
        glycemic_load_enabled: bool = True,
        cooking_coherence_enabled: bool = True,
        gl_all_meals: bool = False,
        temporal_protein_threshold: float = 25.0,
        num_unavailable: int = 8,
    ):
        self.difficulty = difficulty
        self.seed = seed
        self._rng = random.Random(seed)

        if protein_target is None and difficulty in ["medium", "expert"]:
            protein_target = 75.0

        self.constraints = ConstraintConfig(
            difficulty=difficulty,
            calorie_target=calorie_target,
            protein_target=protein_target,
            budget=budget,
            diet_type=diet_type,
            allergies=[a.lower().strip() for a in (allergies or [])],
            ingredient_usage_limits=ingredient_usage_limits or {},
            meal_targets=self._build_meal_targets(difficulty, calorie_target, protein_target),
            glycemic_load_enabled=glycemic_load_enabled,
            cooking_coherence_enabled=cooking_coherence_enabled,
            gl_all_meals=gl_all_meals,
            temporal_protein_threshold=temporal_protein_threshold,
            num_unavailable=num_unavailable,
        )

        self.allowed_ingredients = DIET_COMPATIBILITY[diet_type]
        self.nutrition_db = NUTRITION_DATABASE.copy()
        self.reward_engine = RewardEngine(self.nutrition_db)

        self.episode_id = str(uuid.uuid4())
        self._state: Optional[NutrisyncState] = None

        logger.info(
            f"NutrisyncEnv V2  difficulty={difficulty}, "
            f"calories={calorie_target}, budget={budget}, diet={diet_type}"
        )

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self) -> NutrisyncObservation:
        """Reset environment to initial state, sample seasonal unavailability."""
        self.episode_id = str(uuid.uuid4())

        # Sample unavailable ingredients (seasonal variation)
        k = self.constraints.num_unavailable
        unavailable = set(self._rng.sample(ALL_INGREDIENT_IDS, min(k, len(ALL_INGREDIENT_IDS))))

        self._state = NutrisyncState(
            # Core
            current_meal="breakfast",
            meals_built={"breakfast": [], "lunch": [], "dinner": [], "snack": []},
            step_count=0,
            total_calories_consumed=0.0,
            total_protein_consumed=0.0,
            total_cost_spent=0.0,
            calories_left=self.constraints.calorie_target,
            protein_left=self.constraints.protein_target or 0.0,
            budget_left=self.constraints.budget,
            ingredient_usage_count={},
            constraints=self.constraints,
            allowed_ingredients=self.allowed_ingredients,
            nutrition_db=self.nutrition_db,
            episode_id=self.episode_id,
            seed=self.seed,
            done=False,
            action_history=[],
            reward_history=[],
            # V2 kill switches
            hard_fail=False,
            budget_gate=1.0,
            # V2 satiety
            satiety=50.0,
            accumulated_satiety_penalty=0.0,
            # V2 temporal
            action_space_constrained=False,
            cumulative_protein_g=0.0,
            # V2 clusters
            cluster_usage_counts={},
            all_clusters_used=[],
            all_ingredients_used=[],
            # V2 availability
            availability_violations=0,
            unavailable_ingredients=unavailable,
            # V2 daily totals
            daily_totals={
                "iron_mg": 0.0,
                "calcium_mg": 0.0,
                "fiber_g": 0.0,
                "vitamin_c_mg": 0.0,
                "simple_sugar_g": 0.0,
            },
            daily_calories_consumed=0.0,
            daily_protein_g=0.0,
            daily_carbs_g=0.0,
            daily_fat_g=0.0,
            # V2 step accumulators
            calorie_pacing_total=0.0,
            glycemic_load_total_penalty=0.0,
            meal_completeness_total=0.0,
            cooking_coherence_total=0.0,
        )

        logger.info(f"V2 reset: episode={self.episode_id}, unavailable={len(unavailable)} ings")
        return self._observation()

    # ------------------------------------------------------------------
    # STATE  (required by OpenEnv Environment interface)
    # ------------------------------------------------------------------

    def state(self) -> NutrisyncObservation:
        """Return the current observation (satisfies OpenEnv abstract interface)."""
        if self._state is None:
            return self.reset()
        return self._observation()

    # ------------------------------------------------------------------
    # CLOSE  (required by OpenEnv Environment interface)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources (no-op for this environment)."""
        logger.info("NutrisyncEnv.close() called  no-op.")

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(
        self, action: NutrisyncAction
    ) -> Tuple["NutrisyncObservation", "Reward", bool, Dict[str, Any]]:
        """Execute one environment step (V2)."""
        assert self._state is not None
        s = self._state
        current_meal = s.current_meal
        c = self.constraints

        # ----------------------------------------------------------------
        # PRE-STEP: Satiety window penalty (checked BEFORE processing action)
        # ----------------------------------------------------------------
        if s.step_count > 0:   # only from step 2 onward (after breakfast was played)
            if s.satiety < SATIETY_LOW or s.satiety > SATIETY_HIGH:
                s.accumulated_satiety_penalty += SATIETY_PENALTY
                logger.debug(f"Satiety penalty applied: satiety={s.satiety:.1f}")

        # ----------------------------------------------------------------
        # Handle unavailable ingredients: apply penalty, swap to filler
        # ----------------------------------------------------------------
        processed_items: List[IngredientItem] = []
        for item in action.items:
            if item.ingredient in s.unavailable_ingredients:
                s.availability_violations += 1
                logger.warning(f"Unavailable ingredient selected: {item.ingredient}  swapped to {UNAVAILABLE_FILLER}")
                processed_items.append(IngredientItem(
                    ingredient=UNAVAILABLE_FILLER,
                    quantity=item.quantity,
                    cooking_method=item.cooking_method,
                ))
            else:
                processed_items.append(item)

        # ----------------------------------------------------------------
        # Hard constraint checks (allergy, diet mode, quota)
        # Sets hard_fail = True but continues episode
        # ----------------------------------------------------------------
        valid_items: List[IngredientItem] = []
        for item in processed_items:
            ing = item.ingredient

            # Unknown ingredient  skip silently
            if ing not in self.nutrition_db:
                logger.warning(f"Unknown ingredient skipped: {ing}")
                continue

            # Check allergy
            if ing in c.allergies:
                s.hard_fail = True
                logger.warning(f"HARD FAIL: allergy violation  {ing}")
                continue  # skip this ingredient

            # Check diet mode
            if ing not in DIET_COMPATIBILITY[c.diet_type]:
                s.hard_fail = True
                logger.warning(f"HARD FAIL: diet mode violation  {ing} not allowed for {c.diet_type}")
                continue  # skip this ingredient

            # Check ingredient usage quota
            if c.ingredient_usage_limits:
                current_count = s.ingredient_usage_count.get(ing, 0)
                max_allowed = c.ingredient_usage_limits.get(ing, float("inf"))
                if current_count >= max_allowed:
                    s.hard_fail = True
                    logger.warning(f"HARD FAIL: quota exceeded  {ing} ({current_count}/{int(max_allowed)})")
                    continue  # skip this ingredient

            valid_items.append(item)

        # Use the clean set of valid items
        meal_items = valid_items if valid_items else []

        # ----------------------------------------------------------------
        # Compute nutrition with cooking-method modifiers
        # ----------------------------------------------------------------
        cal, protein, cost, carbs, fat, fiber, v_c, iron, calcium, sugar = \
            self._compute_nutrition_v2(meal_items)

        # ----------------------------------------------------------------
        # Budget gate (per-step, cumulative, irreversible if >3% over)
        # ----------------------------------------------------------------
        projected_spend = s.total_cost_spent + cost
        if s.budget_gate > 0.0:   # only update if not already blown
            overage_pct = (projected_spend - c.budget) / c.budget
            if overage_pct > 0.03:
                s.budget_gate = 0.0   # blown  irreversible
                logger.warning(f"Budget gate BLOWN: overage={overage_pct:.1%}")
            elif overage_pct > 0.0:
                s.budget_gate = min(s.budget_gate, 0.2)   # warning  take worst
                logger.warning(f"Budget gate WARNING: overage={overage_pct:.1%}")

        # ----------------------------------------------------------------
        # Apply meal to state
        # ----------------------------------------------------------------
        s.meals_built[current_meal] = meal_items
        s.total_calories_consumed += cal
        s.total_protein_consumed  += protein
        s.total_cost_spent        += cost
        s.calories_left           -= cal
        s.protein_left            -= protein
        s.budget_left             -= cost

        # V2 daily macro accumulators
        s.daily_calories_consumed += cal
        s.daily_protein_g         += protein
        s.daily_carbs_g           += carbs
        s.daily_fat_g             += fat
        s.cumulative_protein_g    += protein

        # V2 micronutrient totals
        s.daily_totals["iron_mg"]        += iron
        s.daily_totals["calcium_mg"]     += calcium
        s.daily_totals["fiber_g"]        += fiber
        s.daily_totals["vitamin_c_mg"]   += v_c
        s.daily_totals["simple_sugar_g"] += sugar

        # ----------------------------------------------------------------
        # Update satiety AFTER processing meal
        # ----------------------------------------------------------------
        fiber_g_meal  = fiber
        protein_g_meal = protein
        sugar_g_meal   = sugar
        satiety_delta = (fiber_g_meal * 0.8) + (protein_g_meal * 0.4) - (sugar_g_meal * 0.6)
        s.satiety = max(0.0, min(100.0, s.satiety + satiety_delta))

        # ----------------------------------------------------------------
        # Ingredient usage & cluster tracking
        # ----------------------------------------------------------------
        for item in meal_items:
            ing = item.ingredient
            s.ingredient_usage_count[ing] = s.ingredient_usage_count.get(ing, 0) + 1
            s.all_ingredients_used.append(ing)

            # cluster tracking
            cluster = self.nutrition_db[ing].cluster
            s.cluster_usage_counts[cluster] = s.cluster_usage_counts.get(cluster, 0) + 1
            s.all_clusters_used.append(cluster)

        # ----------------------------------------------------------------
        # Per-step V2 scoring accumulators
        # ----------------------------------------------------------------

        # Calorie pacing (non-linear, 5%/10%)
        s.calorie_pacing_total += self._calorie_pacing_score(current_meal, cal, c.calorie_target)

        # Glycemic Load penalty
        if c.glycemic_load_enabled:
            gl_applies = c.gl_all_meals or (current_meal in ["breakfast", "snack"])
            if gl_applies:
                s.glycemic_load_total_penalty += self._glycemic_load_penalty(meal_items)

        # Meal completeness penalty
        s.meal_completeness_total += self._meal_completeness_score(meal_items)

        # Cooking coherence bonus (if enabled)
        if c.cooking_coherence_enabled:
            s.cooking_coherence_total += self._coherence_bonus(meal_items)

        # ----------------------------------------------------------------
        # Advance to next meal
        # ----------------------------------------------------------------
        s.action_history.append(action)
        s.step_count += 1

        current_idx = MEAL_ORDER.index(current_meal)
        if current_idx < len(MEAL_ORDER) - 1:
            s.current_meal = MEAL_ORDER[current_idx + 1]
        else:
            s.done = True

        # ----------------------------------------------------------------
        # Temporal protein-pace constraints for NEXT step
        # ----------------------------------------------------------------
        if not s.done:
            self._check_temporal_constraints(s)

        # ----------------------------------------------------------------
        # Per-step reward (for UI feedback  V2 step signal)
        # ----------------------------------------------------------------
        reward = self.reward_engine.compute(
            meal=current_meal,
            items=meal_items,
            meal_calories=cal,
            meal_protein=protein,
            meal_cost=cost,
            state=s,
            constraints=c,
            is_done=s.done,
        )
        s.reward_history.append(reward)

        # Override with per-step penalty for unavailable ingredient violations
        avail_penalty = sum(
            -0.5 for item in action.items if item.ingredient in s.unavailable_ingredients
        )
        if avail_penalty < 0:
            reward = Reward(
                score=reward.score + avail_penalty,
                breakdown=reward.breakdown,
                feedback=reward.feedback + f" | Unavailability penalty: {avail_penalty:.1f}",
            )

        info = {
            "meal_completed": current_meal,
            "calories_added": cal,
            "protein_added": protein,
            "cost_added": cost,
            "hard_fail": s.hard_fail,
            "budget_gate": s.budget_gate,
            "satiety": s.satiety,
            "reward_details": reward.breakdown.details,
        }

        logger.info(
            f"Step {s.step_count}: {current_meal} done | "
            f"cal={cal:.0f} prot={protein:.1f} cost={cost:.1f} "
            f"satiety={s.satiety:.1f} gate={s.budget_gate} fail={s.hard_fail}"
        )
        return self._observation(), reward, s.done, info

    # ------------------------------------------------------------------
    # V2 HELPER: Compute nutrition with method modifiers
    # ------------------------------------------------------------------

    def _compute_nutrition_v2(
        self, items: List[IngredientItem]
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        """
        Returns (calories, protein, cost, carbs, fat, fiber, vitamin_c, iron, calcium, simple_sugar)
        with cooking-method modifiers applied to calories, protein, vitamin_c, and fiber.
        """
        total_cal = total_prot = total_cost = 0.0
        total_carb = total_fat = total_fiber = total_vc = 0.0
        total_iron = total_ca = total_sugar = 0.0

        for item in items:
            if item.ingredient not in self.nutrition_db:
                continue
            info = self.nutrition_db[item.ingredient]
            scale = item.quantity / 100.0
            mod = METHOD_MODIFIERS.get(item.cooking_method, METHOD_MODIFIERS["boiled"])

            total_cal   += info.calories      * scale * mod["cal"]
            total_prot  += info.protein       * scale * mod["protein"]
            total_cost  += info.cost          * scale
            total_carb  += info.carb_g        * scale
            total_fat   += info.fat_g         * scale
            total_fiber += info.fiber_g       * scale * mod["fiber"]
            total_vc    += info.vitamin_c_mg  * scale * mod["vitamin_c"]
            total_iron  += info.iron_mg       * scale
            total_ca    += info.calcium_mg    * scale
            total_sugar += info.simple_sugar_g * scale

        return (total_cal, total_prot, total_cost, total_carb, total_fat,
                total_fiber, total_vc, total_iron, total_ca, total_sugar)

    # ------------------------------------------------------------------
    # V2 HELPER: Calorie pacing score
    # ------------------------------------------------------------------

    def _calorie_pacing_score(
        self, meal_name: str, actual_cal: float, daily_target: float
    ) -> float:
        target_cal = daily_target * MEAL_CALORIE_FRACTIONS[meal_name]
        if target_cal <= 0:
            return 0.0
        deviation = abs(actual_cal - target_cal) / target_cal
        if deviation <= 0.05:
            return 0.50
        elif deviation <= 0.10:
            return 0.25
        else:
            return 0.0

    # ------------------------------------------------------------------
    # V2 HELPER: Glycemic Load penalty
    # ------------------------------------------------------------------

    def _glycemic_load_penalty(self, items: List[IngredientItem]) -> float:
        GL = 0.0
        for item in items:
            if item.ingredient in self.nutrition_db:
                info = self.nutrition_db[item.ingredient]
                scale = item.quantity / 100.0
                GL += (info.glycemic_index * info.carb_g * scale) / 100.0
        if GL > 20:
            return -0.50
        elif GL > 10:
            return -0.20
        return 0.0

    # ------------------------------------------------------------------
    # V2 HELPER: Meal completeness penalty
    # ------------------------------------------------------------------

    def _meal_completeness_score(self, items: List[IngredientItem]) -> float:
        if not items:
            return -0.15  # empty meal is incomplete
        def in_cluster(clust_set):
            return any(
                self.nutrition_db.get(it.ingredient, NutritionInfo(
                    calories=0, protein=0, cost=0, cluster="vegetables"
                )).cluster in clust_set
                for it in items
                if it.ingredient in self.nutrition_db
            )
        has_protein = in_cluster({"proteins", "legumes", "dairy"})
        has_carbs   = in_cluster({"whole_grains", "refined_carbs"})
        has_fat     = in_cluster({"fats"})
        if all([has_protein, has_carbs, has_fat]):
            return 0.0
        return -0.15

    # ------------------------------------------------------------------
    # V2 HELPER: Cooking coherence bonus
    # ------------------------------------------------------------------

    def _coherence_bonus(self, items: List[IngredientItem]) -> float:
        if not items:
            return 0.0
        optimal_count = sum(
            1 for it in items
            if OPTIMAL_METHODS.get(it.ingredient) == it.cooking_method
        )
        coherence_pct = optimal_count / len(items)
        return 0.30 if coherence_pct >= 0.60 else 0.0

    # ------------------------------------------------------------------
    # V2 HELPER: Temporal protein-pace constraints
    # ------------------------------------------------------------------

    def _check_temporal_constraints(self, s: NutrisyncState) -> None:
        """Filter available ingredients for next step if behind protein pace."""
        if not s.constraints.protein_target:
            s.action_space_constrained = False
            return

        next_meal = s.current_meal
        expected_pct = PROTEIN_PACE_TARGETS.get(next_meal, 1.0)
        expected_protein = s.constraints.protein_target * expected_pct
        deficit = expected_protein - s.cumulative_protein_g

        if deficit > s.constraints.temporal_protein_threshold:
            s.action_space_constrained = True
        else:
            s.action_space_constrained = False

    def _get_available_ingredients(self, s: NutrisyncState) -> List[str]:
        """Return filtered ingredient list for current state."""
        base = [
            ing for ing in self.allowed_ingredients
            if ing not in s.unavailable_ingredients
        ]
        if s.action_space_constrained and s.constraints.protein_target:
            # Filter: only ingredients with  MIN_PROTEIN_PER_ING per 100g, or essential groups for completeness
            base = [
                ing for ing in base
                if self.nutrition_db[ing].protein >= MIN_PROTEIN_PER_ING
                # Preserve basic grains, vegetables, and fats to satisfy meal completeness
                or self.nutrition_db[ing].cluster in {"whole_grains", "refined_carbs", "fats", "vegetables", "leafy_greens"}
            ]
        return sorted(base)

    def _get_protein_pace_deficit(self, s: NutrisyncState) -> float:
        if not s.constraints.protein_target:
            return 0.0
        next_meal = s.current_meal
        expected_pct = PROTEIN_PACE_TARGETS.get(next_meal, 1.0)
        expected = s.constraints.protein_target * expected_pct
        return max(0.0, expected - s.cumulative_protein_g)

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def _observation(self) -> NutrisyncObservation:
        s = self._state
        assert s is not None

        available = self._get_available_ingredients(s)
        deficit = self._get_protein_pace_deficit(s)

        return NutrisyncObservation(
            current_meal=s.current_meal,
            calories_left=max(0.0, s.calories_left),
            protein_left=max(0.0, s.protein_left),
            budget_left=max(0.0, s.budget_left),
            meals_built={
                meal: [
                    {"ingredient": it.ingredient, "quantity": it.quantity,
                     "cooking_method": it.cooking_method}
                    for it in items
                ]
                for meal, items in s.meals_built.items()
            },
            ingredient_usage_count=s.ingredient_usage_count.copy(),
            allowed_ingredients=sorted(self.allowed_ingredients),
            constraints={
                "difficulty": s.constraints.difficulty,
                "diet_type":  s.constraints.diet_type,
                "allergies":  s.constraints.allergies,
                "calorie_target": s.constraints.calorie_target,
                "protein_target": s.constraints.protein_target,
                "budget": s.constraints.budget,
            },
            # V2 fields
            satiety=s.satiety,
            cumulative_protein_g=s.cumulative_protein_g,
            protein_pace_deficit=deficit,
            available_ingredients=available,
            unavailable_ingredients=sorted(s.unavailable_ingredients),
            daily_totals=dict(s.daily_totals),
            cluster_usage_counts=dict(s.cluster_usage_counts),
            action_space_constrained=s.action_space_constrained,
            budget_gate=s.budget_gate,
            hard_fail=s.hard_fail,
            reward=s.reward_history[-1].score if s.reward_history else 0.0,
            done=s.done,
            metadata={
                "step_count": s.step_count,
                "availability_violations": s.availability_violations,
            },
        )

    # ------------------------------------------------------------------
    # Build meal targets
    # ------------------------------------------------------------------

    def _build_meal_targets(
        self,
        difficulty: Literal["easy", "medium", "expert"],
        calorie_target: float,
        protein_target: Optional[float],
    ) -> Optional[Dict[str, MealTarget]]:
        if difficulty not in ["medium", "expert"]:
            return None
        cal = calorie_target
        prot = protein_target or 0
        return {
            "breakfast": MealTarget(
                min_calories=cal * 0.22, max_calories=cal * 0.28,
                min_protein=prot * 0.20 if prot else None,
                max_protein=prot * 0.30 if prot else None,
            ),
            "lunch": MealTarget(
                min_calories=cal * 0.32, max_calories=cal * 0.38,
                min_protein=prot * 0.30 if prot else None,
                max_protein=prot * 0.40 if prot else None,
            ),
            "dinner": MealTarget(
                min_calories=cal * 0.27, max_calories=cal * 0.33,
                min_protein=prot * 0.25 if prot else None,
                max_protein=prot * 0.35 if prot else None,
            ),
            "snack": MealTarget(
                min_calories=cal * 0.08, max_calories=cal * 0.12,
                min_protein=prot * 0.05 if prot else None,
                max_protein=prot * 0.15 if prot else None,
            ),
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_state(self) -> NutrisyncState:
        assert self._state is not None
        return self._state

    def get_episode_summary(self) -> Dict[str, Any]:
        assert self._state is not None
        s = self._state
        return {
            "episode_id": self.episode_id,
            "difficulty": self.difficulty,
            "total_steps": s.step_count,
            "total_calories": s.total_calories_consumed,
            "total_protein": s.total_protein_consumed,
            "total_cost": s.total_cost_spent,
            "target_calories": self.constraints.calorie_target,
            "target_protein": self.constraints.protein_target,
            "budget": self.constraints.budget,
            "meals_built": len([m for m in s.meals_built.values() if m]),
            "done": s.done,
            "hard_fail": s.hard_fail,
            "budget_gate": s.budget_gate,
            "satiety": s.satiety,
        }
