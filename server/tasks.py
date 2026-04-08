"""
NutriSync V2 Task Definitions and Graders.

Three tasks with progressively harder configurations:
  - easy:   Omnivore, Rs.250, 2000 cal, GL + coherence disabled
  - medium: Vegetarian, Rs.200, 2300 cal, coherence disabled
  - hard:   Vegan, Rs.160, 2300 cal, all penalties active, GL all meals

Graders delegate to `compute_episode_reward()` and scale to [0, 1].
"""

from typing import Any, Callable, Dict
import logging

try:
    from models import NutrisyncState
except ImportError:
    try:
        from NutriSync.models import NutrisyncState
    except ImportError:
        from ..models import NutrisyncState

try:
    from reward import compute_episode_reward
except ImportError:
    from .reward import compute_episode_reward

logger = logging.getLogger(__name__)


# ============================================================================
# V2 GRADERS  (score  [0, 1]  divide episode reward by 10)
# ============================================================================

def grade_easy(state: NutrisyncState) -> float:
    """
    Easy grader  delegates to V2 episode reward pipeline.
    Max achievable: ~7.5/10  ~0.75.
    Glycemic load and cooking coherence are disabled at this tier.
    """
    if not state.done or state.step_count < 4:
        return 0.0
    raw = compute_episode_reward(state)   # 010 scale
    normalized = raw / 10.0
    logger.info(f"Grade easy: raw={raw:.2f} normalized={normalized:.3f}")
    return round(normalized, 4)


def grade_medium(state: NutrisyncState) -> float:
    """
    Medium grader  delegates to V2 episode reward pipeline.
    Max achievable: ~6.5/10  ~0.65.
    Vegetarian + tighter budget + micronutrients. Cooking coherence disabled.
    """
    if not state.done or state.step_count < 4:
        return 0.0
    raw = compute_episode_reward(state)
    normalized = raw / 10.0
    logger.info(f"Grade medium: raw={raw:.2f} normalized={normalized:.3f}")
    return round(normalized, 4)


def grade_hard(state: NutrisyncState) -> float:
    """
    Hard grader  delegates to V2 episode reward pipeline.
    Max achievable: ~5.5/10  ~0.55.
    Vegan + tight budget + 120g protein + all penalties active including GL all meals.
    """
    if not state.done or state.step_count < 4:
        return 0.0
    raw = compute_episode_reward(state)
    normalized = raw / 10.0
    logger.info(f"Grade hard: raw={raw:.2f} normalized={normalized:.3f}")
    return round(normalized, 4)


# ============================================================================
# TASK REGISTRY
# ============================================================================

TASKS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # EASY: Omnivore, Rs.250, 2000 cal, 100g protein
    #   - GL penalty: disabled
    #   - Cooking coherence: disabled
    #   - Seasonal: 8 unavailable ingredients
    #   - Max achievable score: ~7.5/10
    # ------------------------------------------------------------------
    "easy": {
        "difficulty": "easy",
        "calorie_target": 2000,
        "protein_target": 100.0,
        "budget": 250,
        "diet_type": "omnivore",
        "allergies": [],
        "ingredient_usage_limits": None,
        # V2 feature flags
        "glycemic_load_enabled": False,
        "cooking_coherence_enabled": False,
        "gl_all_meals": False,
        "temporal_protein_threshold": 25.0,
        "num_unavailable": 8,
        "grader": grade_easy,
    },

    # ------------------------------------------------------------------
    # MEDIUM: Vegetarian, Rs.200, 2300 cal, 100g protein
    #   - GL penalty: active (breakfast + snack only)
    #   - Cooking coherence: disabled
    #   - Micronutrient floors: iron + calcium + fiber
    #   - Seasonal: 10 unavailable ingredients
    #   - Max achievable score: ~6.5/10
    # ------------------------------------------------------------------
    "medium": {
        "difficulty": "medium",
        "calorie_target": 2300,
        "protein_target": 100.0,
        "budget": 200,
        "diet_type": "vegetarian",
        "allergies": [],
        "ingredient_usage_limits": None,
        # V2 feature flags
        "glycemic_load_enabled": True,
        "cooking_coherence_enabled": False,
        "gl_all_meals": False,
        "temporal_protein_threshold": 25.0,
        "num_unavailable": 10,
        "grader": grade_medium,
    },

    # ------------------------------------------------------------------
    # HARD: Vegan, Rs.160, 2300 cal, 120g protein
    #   - All penalties fully active
    #   - GL enforced at ALL 4 meals (not just breakfast + snack)
    #   - Cooking coherence: active
    #   - Tighter temporal threshold: 20g (not 25g)
    #   - All 4 micronutrient floors active
    #   - Seasonal: 12 unavailable ingredients
    #   - Max achievable score: ~5.5/10
    # ------------------------------------------------------------------
    "hard": {
        "difficulty": "expert",
        "calorie_target": 2300,
        "protein_target": 120.0,
        "budget": 160,
        "diet_type": "vegan",
        "allergies": ["coconut", "coconut_oil"],
        "ingredient_usage_limits": {"rice": 2, "mustard_oil": 2},
        # V2 feature flags
        "glycemic_load_enabled": True,
        "cooking_coherence_enabled": True,
        "gl_all_meals": True,          # GL applied to ALL 4 meals in hard
        "temporal_protein_threshold": 20.0,   # stricter threshold
        "num_unavailable": 12,
        "grader": grade_hard,
    },
}
