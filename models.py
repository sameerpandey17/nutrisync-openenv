# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Production-grade Pydantic models for the Nutrisync RL Environment V2.

V2 adds: fiber, simple sugar, micronutrients, glycemic index, cluster tags,
cooking methods, satiety state, budget gate, hard-fail kill switch,
temporal coupling, and seasonal availability.
"""

from typing import Any, Dict, List, Literal, Optional, Set
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# NUTRITION & COST DATA STRUCTURES (V2 Extended)
# ============================================================================

class NutritionInfo(BaseModel):
    """Nutrition information per 100g of an ingredient (V2 extended)."""

    # Core macros
    calories: float = Field(..., ge=0, description="Calories per 100g")
    protein: float = Field(..., ge=0, description="Protein (g) per 100g")
    cost: float = Field(..., ge=0, description="Cost (INR) per 100g")

    # V2: Macros for ratio scoring
    carb_g: float = Field(default=0.0, ge=0, description="Carbohydrates (g) per 100g")
    fat_g: float = Field(default=0.0, ge=0, description="Fat (g) per 100g")

    # V2: Satiety components
    fiber_g: float = Field(default=0.0, ge=0, description="Dietary fiber (g) per 100g")
    simple_sugar_g: float = Field(default=0.0, ge=0, description="Simple sugars (g) per 100g")

    # V2: Micronutrients
    iron_mg: float = Field(default=0.0, ge=0, description="Iron (mg) per 100g")
    calcium_mg: float = Field(default=0.0, ge=0, description="Calcium (mg) per 100g")
    vitamin_c_mg: float = Field(default=0.0, ge=0, description="Vitamin C (mg) per 100g")

    # V2: Glycemic index (0 = not applicable for pure fats/proteins)
    glycemic_index: int = Field(default=0, ge=0, le=100, description="Glycemic Index (0-100)")

    # V2: Semantic cluster tag
    cluster: str = Field(default="vegetables", description="Semantic food cluster")


class IngredientItem(BaseModel):
    """A single ingredient with quantity and cooking method in a meal."""

    ingredient: str = Field(..., description="Ingredient name (ID in NUTRITION_DATABASE)")
    quantity: float = Field(..., gt=0, description="Quantity in grams")
    # V2: cooking method  defaults to 'boiled' for backward compatibility
    cooking_method: Literal[
        "raw", "boiled", "steamed", "sauteed", "fried", "roasted", "fermented"
    ] = Field(default="boiled", description="Cooking method applied to ingredient")

    @field_validator("ingredient")
    @classmethod
    def validate_ingredient_name(cls, v: str) -> str:
        """Ensure ingredient name is non-empty."""
        if not v or not v.strip():
            raise ValueError("Ingredient name cannot be empty")
        return v.lower().strip()


from openenv.core.env_server.types import Observation as OpenEnvObservation
from openenv.core.env_server.types import State as OpenEnvState


# ============================================================================
# ACTION MODEL
# ============================================================================

class NutrisyncAction(BaseModel):
    """Agent action: select ingredients for current meal."""

    items: List[IngredientItem] = Field(
        default_factory=list,
        description="List of ingredients, quantities, and cooking methods to add to meal"
    )


# ============================================================================
# OBSERVATION MODEL (V2 Extended)
# ============================================================================

class NutrisyncObservation(OpenEnvObservation):
    """Environment observation returned to agent (V2 extended)."""

    current_meal: Literal["breakfast", "lunch", "dinner", "snack", "done"] = Field(
        ..., description="Current meal being built. 'done' if episode is finished."
    )
    calories_left: float = Field(..., ge=0, description="Remaining daily calorie budget")
    protein_left: float = Field(..., ge=0, description="Remaining daily protein budget")
    budget_left: float = Field(..., ge=0, description="Remaining money budget (INR)")
    meals_built: Dict[str, List[Dict[str, Any]]] = Field(
        ..., description="Meals already constructed"
    )
    ingredient_usage_count: Dict[str, int] = Field(
        ..., description="Number of times each ingredient used across meals"
    )
    allowed_ingredients: List[str] = Field(
        ..., description="All diet-compatible ingredients (unfiltered)"
    )
    constraints: Dict[str, Any] = Field(
        ..., description="Active constraints (difficulty, diet_type, allergies, targets, etc.)"
    )

    # V2: Satiety and temporal coupling
    satiety: float = Field(default=50.0, description="Current satiety level [0-100]. Target window: [40, 80].")
    cumulative_protein_g: float = Field(default=0.0, description="Total protein consumed so far (g)")
    protein_pace_deficit: float = Field(default=0.0, description="Grams behind protein pace target (positive = behind)")

    # V2: Filtered action space
    available_ingredients: List[str] = Field(
        default_factory=list,
        description="Filtered available ingredients for this step (respects temporal + seasonal constraints)"
    )
    unavailable_ingredients: List[str] = Field(
        default_factory=list,
        description="Seasonally unavailable ingredients for this episode"
    )

    # V2: Running micronutrient totals
    daily_totals: Dict[str, float] = Field(
        default_factory=dict,
        description="Running daily totals: iron_mg, calcium_mg, fiber_g, vitamin_c_mg, simple_sugar_g"
    )

    # V2: Cluster tracking
    cluster_usage_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of ingredient-uses per semantic cluster (for spam detection)"
    )

    # V2: State flags
    action_space_constrained: bool = Field(
        default=False,
        description="True if temporal protein constraint is active (only high-protein options available)"
    )
    budget_gate: float = Field(
        default=1.0,
        description="Current budget gate multiplier. 1.0=clean, 0.1=warning, 0.0=blown (irreversible)"
    )
    hard_fail: bool = Field(
        default=False,
        description="True if a hard constraint was violated; episode reward will be 0.0"
    )

    # OpenEnvObservation provides: `done`, `reward`, `metadata`


# ============================================================================
# REWARD MODEL
# ============================================================================

class RewardBreakdown(BaseModel):
    """Detailed breakdown of reward components."""

    goal_progress: float = Field(default=0.0, description="Progress toward nutritional goals")
    constraint_penalty: float = Field(default=0.0, description="Penalty for constraint violations")
    efficiency_bonus: float = Field(default=0.0, description="Bonus for budget efficiency")
    composition_bonus: float = Field(default=0.0, description="Bonus for meal composition quality")
    details: Dict[str, float] = Field(
        default_factory=dict,
        description="Fine-grained per-component scores"
    )


class Reward(BaseModel):
    """Reward signal returned after each step."""

    score: float = Field(..., description="Step reward score")
    breakdown: RewardBreakdown = Field(
        default_factory=RewardBreakdown,
        description="Detailed reward components"
    )
    feedback: str = Field(..., description="Human-readable feedback message")


# ============================================================================
# CONSTRAINT & MEAL TARGET MODELS
# ============================================================================

class MealTarget(BaseModel):
    """Target ranges for a meal (used in MEDIUM/EXPERT modes)."""

    min_calories: float = Field(..., ge=0, description="Minimum calories")
    max_calories: float = Field(..., ge=0, description="Maximum calories")
    min_protein: Optional[float] = Field(default=None, description="Minimum protein (g)")
    max_protein: Optional[float] = Field(default=None, description="Maximum protein (g)")


class ConstraintConfig(BaseModel):
    """Configuration of active constraints for an episode."""

    difficulty: Literal["easy", "medium", "expert"] = Field(
        ..., description="Difficulty level"
    )
    calorie_target: float = Field(..., gt=0, description="Total daily calorie target")
    protein_target: Optional[float] = Field(
        default=None, description="Total daily protein target (g)"
    )
    budget: float = Field(..., gt=0, description="Total budget (INR)")
    diet_type: Literal["omnivore", "vegetarian", "vegan"] = Field(
        ..., description="Dietary mode  drives hard constraint enforcement"
    )
    allergies: List[str] = Field(
        default_factory=list, description="Allergenic ingredients (hard constraint)"
    )
    ingredient_usage_limits: Optional[Dict[str, int]] = Field(
        default=None, description="Max times each ingredient can be used across day (hard quota)"
    )
    meal_targets: Optional[Dict[str, MealTarget]] = Field(
        default=None, description="Per-meal nutritional targets"
    )

    # V2: Task-level feature flags
    glycemic_load_enabled: bool = Field(
        default=True, description="Apply GL penalty at breakfast and snack"
    )
    cooking_coherence_enabled: bool = Field(
        default=True, description="Award cooking coherence bonus per meal"
    )
    gl_all_meals: bool = Field(
        default=False, description="Apply GL penalty to ALL meals (hard mode only)"
    )
    temporal_protein_threshold: float = Field(
        default=25.0, description="Grams behind protein pace that triggers action-space filtering"
    )
    num_unavailable: int = Field(
        default=8, description="Number of ingredients randomly marked unavailable at episode start"
    )


# ============================================================================
# INTERNAL STATE MODEL (V2 Extended)
# ============================================================================

class NutrisyncState(OpenEnvState):
    """Full internal state of the environment (V2 extended)."""

    # Meal tracking
    current_meal: Literal["breakfast", "lunch", "dinner", "snack", "done"]
    meals_built: Dict[str, List[IngredientItem]]

    # Legacy nutrition tracking (kept for grader backward-compat)
    total_calories_consumed: float
    total_protein_consumed: float
    total_cost_spent: float

    # Remaining budgets
    calories_left: float
    protein_left: float
    budget_left: float

    # Ingredient tracking
    ingredient_usage_count: Dict[str, int]

    # Configuration
    constraints: ConstraintConfig
    allowed_ingredients: List[str]
    nutrition_db: Dict[str, NutritionInfo]

    # Episode metadata
    seed: Optional[int]
    done: bool

    # Action history
    action_history: List[NutrisyncAction] = Field(default_factory=list)
    reward_history: List[Reward] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # V2: Kill switches
    # -----------------------------------------------------------------------
    hard_fail: bool = Field(
        default=False,
        description="Irreversible. True  episode_reward = 0.0"
    )
    budget_gate: float = Field(
        default=1.0,
        description="Multiplicative gate. 1.0=clean, 0.1=warning, 0.0=blown (irreversible)"
    )

    # -----------------------------------------------------------------------
    # V2: Satiety persistent state
    # -----------------------------------------------------------------------
    satiety: float = Field(default=50.0, description="Satiety level [0-100], starts at 50")
    accumulated_satiety_penalty: float = Field(
        default=0.0, description="Accumulated satiety window penalties (min -0.9)"
    )

    # -----------------------------------------------------------------------
    # V2: Temporal coupling
    # -----------------------------------------------------------------------
    action_space_constrained: bool = Field(default=False)
    cumulative_protein_g: float = Field(default=0.0)

    # -----------------------------------------------------------------------
    # V2: Cluster tracking
    # -----------------------------------------------------------------------
    cluster_usage_counts: Dict[str, int] = Field(default_factory=dict)
    all_clusters_used: List[str] = Field(default_factory=list)
    all_ingredients_used: List[str] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # V2: Seasonal availability
    # -----------------------------------------------------------------------
    availability_violations: int = Field(default=0)
    unavailable_ingredients: Set[str] = Field(default_factory=set)

    # -----------------------------------------------------------------------
    # V2: Micronutrient & macro daily accumulators
    # -----------------------------------------------------------------------
    daily_totals: Dict[str, float] = Field(
        default_factory=lambda: {
            "iron_mg": 0.0,
            "calcium_mg": 0.0,
            "fiber_g": 0.0,
            "vitamin_c_mg": 0.0,
            "simple_sugar_g": 0.0,
        }
    )
    daily_calories_consumed: float = Field(default=0.0)
    daily_protein_g: float = Field(default=0.0)
    daily_carbs_g: float = Field(default=0.0)
    daily_fat_g: float = Field(default=0.0)

    # -----------------------------------------------------------------------
    # V2: Per-step accumulators (read by compute_episode_reward)
    # -----------------------------------------------------------------------
    calorie_pacing_total: float = Field(default=0.0)
    glycemic_load_total_penalty: float = Field(default=0.0)
    meal_completeness_total: float = Field(default=0.0)
    cooking_coherence_total: float = Field(default=0.0)


# ============================================================================
# STEP RESULT
# ============================================================================

class StepResult(BaseModel):
    """Result of a single environment step."""

    observation: NutrisyncObservation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
