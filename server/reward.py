# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NutriSync V2 Reward Engine.

Implements an 8-stage multiplicative+additive pipeline that forces genuine
multi-step reasoning and eliminates all known V1 exploit paths.

Stage 1: Hard gates (kill switches)
Stage 2: Budget gate (multiplicative, irreversible)
Stage 3: Cluster spam penalty (multiplicative, stacking)
Stage 4: Core nutritional scoring (calorie pacing + macro ratios)
Stage 5: Behavioral penalties (satiety, glycemic load, meal completeness)
Stage 6: Action quality bonuses (cooking coherence, availability)
Stage 7: Episode bonuses (variety, budget efficiency, clean-run)
Stage 8: Normalize to [0, 1]

Backward-compatible `compute()` method is preserved for per-step UI feedback.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from models import (
    IngredientItem,
    NutritionInfo,
    Reward,
    RewardBreakdown,
    ConstraintConfig,
    NutrisyncState,
)

logger = logging.getLogger(__name__)


# ============================================================================
# V2 CONSTANTS
# ============================================================================

TARGET_RATIOS = {
    "protein": (0.20, 0.35),   # 2035% of total calories from protein
    "carbs":   (0.40, 0.55),   # 4055% of total calories from carbs
    "fat":     (0.20, 0.30),   # 2030% of total calories from fat
}

MICRONUTRIENT_FLOORS = {
    "iron_mg":      15.0,
    "calcium_mg":   600.0,
    "fiber_g":      25.0,
    "vitamin_c_mg": 65.0,
}

# Food groups for V1 backward-compat helpers
FOOD_GROUPS: Dict[str, str] = {
    "chicken": "PROTEIN",   "mutton": "PROTEIN",    "fish_rohu": "PROTEIN",
    "eggs": "PROTEIN",      "paneer": "PROTEIN",     "tofu": "PROTEIN",
    "moong_dal": "PROTEIN", "toor_dal": "PROTEIN",   "chana_dal": "PROTEIN",
    "masoor_dal": "PROTEIN","rajma": "PROTEIN",      "chickpeas": "PROTEIN",
    "rice": "GRAIN",        "wheat_flour": "GRAIN",  "roti": "GRAIN",
    "oats": "GRAIN",        "poha": "GRAIN",         "semolina": "GRAIN",
    "bread": "GRAIN",       "millet_bajra": "GRAIN", "ragi": "GRAIN",
    "potato": "VEGETABLE",  "onion": "VEGETABLE",    "tomato": "VEGETABLE",
    "spinach": "VEGETABLE", "cauliflower": "VEGETABLE","cabbage": "VEGETABLE",
    "brinjal": "VEGETABLE", "lady_finger": "VEGETABLE","bottle_gourd": "VEGETABLE",
    "carrot": "VEGETABLE",  "green_peas": "VEGETABLE","capsicum": "VEGETABLE",
    "banana": "FRUIT",      "apple": "FRUIT",        "mango": "FRUIT",
    "papaya": "FRUIT",      "guava": "FRUIT",        "coconut": "FRUIT",
    "milk": "DAIRY",        "curd": "DAIRY",         "ghee": "DAIRY",
    "butter": "DAIRY",      "cheese": "DAIRY",
    "mustard_oil": "FAT_CONDIMENT",  "coconut_oil": "FAT_CONDIMENT",
    "groundnut_oil": "FAT_CONDIMENT","sugar": "FAT_CONDIMENT",
    "jaggery": "FAT_CONDIMENT",      "honey": "FAT_CONDIMENT",
}

REALISTIC_PORTIONS: Dict[str, tuple] = {
    "PROTEIN": (50, 250), "GRAIN": (50, 250), "VEGETABLE": (50, 300),
    "FRUIT": (50, 250), "DAIRY": (50, 300), "FAT_CONDIMENT": (5, 30),
}

MEAL_CAL_FRACTIONS: Dict[str, tuple] = {
    "breakfast": (0.22, 0.28), "lunch": (0.32, 0.38),
    "dinner": (0.27, 0.33),    "snack": (0.08, 0.12),
}

HIGH_DENSITY_ITEMS: Set[str] = {
    "ghee", "butter", "mustard_oil", "coconut_oil", "groundnut_oil",
    "cheese", "sugar", "honey",
}


# ============================================================================
# V2 EPISODE REWARD PIPELINE
# ============================================================================

def compute_episode_reward(state: NutrisyncState) -> float:
    """
    Full V2 PUNISHING episode reward pipeline.
    Returns a score in [0, 10].

    Strict Build-From-Zero approach. Points are only awarded for hitting 
    perfect constraints. Offenses apply heavy multipliers against the earned score.
    """

    # ------------------------------------------------------------------
    # Stage 1: Hard gates  kill switches checked first
    # ------------------------------------------------------------------
    if state.hard_fail:
        logger.info("Episode reward: 0.0 (hard_fail)")
        return 0.0

    # START FROM ZERO
    reward = 0.0

    # ------------------------------------------------------------------
    # Stage 2: Core nutritional scoring (additive to base)
    # ------------------------------------------------------------------

    # Calorie pacing (max +2.0)
    reward += max(0.0, state.calorie_pacing_total)

    # Macro ratio score (max +3.0)
    reward += _macro_ratio_score(state)

    # Micronutrient score (max +2.0)
    reward += _micronutrient_score(state)

    # ------------------------------------------------------------------
    # Stage 3: Action quality & Episode bonuses (additive to base)
    # ------------------------------------------------------------------

    # Cooking coherence bonus (max +1.2)
    reward += max(0.0, state.cooking_coherence_total)

    # Variety (max +0.8) and Budget (max +0.7) -> total +1.5
    reward += _variety_bonus(state)           
    reward += _budget_efficiency_bonus(state)

    # Clean run bonus (+3.0 buffer) if no availability violations
    if state.availability_violations == 0:
        reward += 3.0

    # Theoretical Maximum Base Score = 11.0 (will be capped)

    # ------------------------------------------------------------------
    # Stage 4: Fatal Gate
    # ------------------------------------------------------------------
    reward *= state.budget_gate
    
    if reward <= 0.0:
        logger.info("Episode reward: 0.0 (budget_gate blown)")
        return 0.0

    # ------------------------------------------------------------------
    # Stage 5: Aggressive Multiplicative Penalties (Tuned to achieve targets)
    # ------------------------------------------------------------------
    
    # 5a. Cluster spam penalty
    spam_multiplier = _compute_cluster_spam_multiplier(state)
    reward *= spam_multiplier

    # 5b. Satiety Multiplier
    # Each out-of-bounds step was approx -0.3 in the state accumulator. 
    # Multiply score by 0.95 per missed step (loosened from 0.90)
    satiety_misses = round(abs(state.accumulated_satiety_penalty) / 0.3) if state.accumulated_satiety_penalty < 0 else 0
    reward *= (0.95 ** satiety_misses)

    # 5c. Glycemic Load Multiplier
    # Each bad GL meal accumulates -0.2 to -0.5. 
    # Multiply score by 0.98 per ~0.20 of GL penalty (loosened from 0.95)
    gl_misses = abs(state.glycemic_load_total_penalty) / 0.20 if state.glycemic_load_total_penalty < 0 else 0
    reward *= (0.98 ** gl_misses)

    # 5d. Meal completeness Multiplier
    # Incomplete meals gave -0.15. 
    # Multiply score by 0.95 per missed step (loosened from 0.85)
    completeness_misses = abs(state.meal_completeness_total) / 0.15 if state.meal_completeness_total < 0 else 0
    reward *= (0.95 ** completeness_misses)
    
    # 5e. Availability Violation Multiplier
    reward *= (0.75 ** state.availability_violations)

    # ------------------------------------------------------------------
    # Stage 8: Normalize to [0, 1]
    # ------------------------------------------------------------------
    reward = max(0.0, min(10.0, reward)) / 10.0   # scale [0,10] → [0,1]
    
    with open("reward_debug.log", "a", encoding="utf-8") as _f:
        _f.write(f"[REWARD DEBUG] spam:{spam_multiplier:.2f} sat:{satiety_misses} gl:{gl_misses} comp:{completeness_misses} avail:{state.availability_violations} -> {reward:.4f}\n")
        
    logger.info(f"Episode reward (V2 Punishing, [0,1]): {reward:.4f}")
    return round(reward, 4)


# ============================================================================
# STAGE-SPECIFIC HELPERS
# ============================================================================

def _compute_cluster_spam_multiplier(state: NutrisyncState) -> float:
    """Stage 3: cluster-specific spam limits to support variety without breaking."""
    multiplier = 1.0
    # Increase allowances for common meal bases
    cluster_limits = {
        "vegetables": 8,
        "fats": 6,
        "whole_grains": 6,
        "proteins": 4,
        "dairy": 4,
        "legumes": 4,
        "refined_carbs": 3,
    }
    for cluster, count in state.cluster_usage_counts.items():
        limit = cluster_limits.get(cluster, 2)
        if count > limit:
            excess = count - limit
            multiplier *= (0.85 ** excess)  # softened from 0.70
    return multiplier


def _macro_ratio_score(state: NutrisyncState) -> float:
    """Stage 4b: Max +3.0 (+1.0 per macro in range)."""
    total_cal = state.daily_calories_consumed
    if total_cal <= 0:
        return 0.0

    p_pct = (state.daily_protein_g * 4) / total_cal
    c_pct = (state.daily_carbs_g   * 4) / total_cal
    f_pct = (state.daily_fat_g     * 9) / total_cal

    in_range = [
        TARGET_RATIOS["protein"][0] <= p_pct <= TARGET_RATIOS["protein"][1],
        TARGET_RATIOS["carbs"][0]   <= c_pct <= TARGET_RATIOS["carbs"][1],
        TARGET_RATIOS["fat"][0]     <= f_pct <= TARGET_RATIOS["fat"][1],
    ]
    return sum(in_range) * 1.0   # max +3.0


def _micronutrient_score(state: NutrisyncState) -> float:
    """Stage 4c: Max +2.0. Gives +0.5 per nutrient floor achieved."""
    score = 0.0
    totals = state.daily_totals

    for nutrient, floor in MICRONUTRIENT_FLOORS.items():
        if floor > 0:
            actual = totals.get(nutrient, 0.0)
            if actual >= floor:
                score += 0.50

    return score   # range: 0.0 to +2.0


def _variety_bonus(state: NutrisyncState) -> float:
    """Stage 7a: Max +0.8."""
    unique_ings     = len(set(state.all_ingredients_used))
    unique_clusters = len(set(state.all_clusters_used))

    if unique_ings >= 12 and unique_clusters >= 4:
        return 0.8
    elif unique_ings >= 8 and unique_clusters >= 3:
        return 0.4
    return 0.0


def _budget_efficiency_bonus(state: NutrisyncState) -> float:
    """Stage 7b: Max +0.7."""
    budget = state.constraints.budget
    if budget <= 0:
        return 0.0
    utilization = state.total_cost_spent / budget

    if 0.85 <= utilization <= 0.95:
        return 0.7
    elif 0.70 <= utilization < 0.85:
        return 0.3
    return 0.0


# ============================================================================
# REWARD ENGINE CLASS (backward-compatible per-step feedback)
# ============================================================================

class RewardEngine:
    """
    Multi-component per-step reward engine (V2 backward-compatible).

    Per-step `compute()` provides dense UI feedback.
    Episode-end scoring is done by `compute_episode_reward()` above.
    """

    def __init__(self, nutrition_db: Dict[str, NutritionInfo]):
        self.nutrition_db = nutrition_db

    # ------------------------------------------------------------------ #
    #  PUBLIC API  per-step dense reward for UI feedback                 #
    # ------------------------------------------------------------------ #

    def compute(
        self,
        meal: str,
        items: List[IngredientItem],
        meal_calories: float,
        meal_protein: float,
        meal_cost: float,
        state: NutrisyncState,
        constraints: ConstraintConfig,
        is_done: bool,
    ) -> Reward:
        """Compute per-step reward for UI feedback."""
        d: Dict[str, float] = {}
        fb: List[str] = [f"Meal: {meal}"]

        # ---- Tier 1: Per-Meal Scores ----
        d["calorie_proximity"] = self._calorie_proximity(meal, meal_calories, constraints)
        d["protein_proximity"] = (
            self._protein_proximity(meal, meal_protein, constraints)
            if constraints.difficulty in ("medium", "expert") and constraints.protein_target
            else 0.0
        )
        d["cost_pacing"]          = self._cost_pacing(meal, meal_cost, constraints.budget)
        d["ingredient_diversity"] = self._ingredient_diversity(items)
        d["food_group_coverage"]  = self._food_group_coverage(items)
        d["portion_realism"]      = self._portion_realism(items)

        # ---- Tier 2: Anti-Gaming Penalties ----
        d["repetition_penalty"]    = self._repetition_penalty(items, state, meal)
        d["empty_meal_penalty"]    = -0.5 if not items else 0.0
        d["calorie_dense_penalty"] = self._calorie_dense_penalty(items, meal_calories)

        # ---- V2 add-ons for step signal ----
        d["satiety_state"] = (
            -0.3 if (state.satiety < 40.0 or state.satiety > 80.0) else 0.0
        )
        d["budget_gate_signal"] = 0.0 if state.budget_gate == 1.0 else (
            -2.0 if state.budget_gate == 0.0 else -1.0
        )
        d["hard_fail_signal"] = -5.0 if state.hard_fail else 0.0

        # ---- Tier 3: Episode completion bonuses (final step only) ----
        if is_done:
            d["final_calorie_accuracy"] = self._final_calorie_accuracy(state, constraints)
            d["final_protein_accuracy"] = (
                self._final_protein_accuracy(state, constraints)
                if constraints.difficulty in ("medium", "expert") and constraints.protein_target
                else 0.0
            )
            d["budget_utilization"] = self._budget_utilization(state, constraints)
            d["global_diversity"]   = self._global_diversity(state)
            d["meal_balance"]       = self._meal_balance(state, constraints)
            d["cross_meal_variety"] = self._cross_meal_variety(state)
        else:
            for k in ("final_calorie_accuracy", "final_protein_accuracy",
                      "budget_utilization", "global_diversity",
                      "meal_balance", "cross_meal_variety"):
                d[k] = 0.0

        # ---- Aggregate ----
        W = {"cal": 1.0, "prot": 0.8, "cost": 0.4, "div": 0.5, "grp": 0.5, "portion": 0.3}

        goal_progress = (
            d["calorie_proximity"] * W["cal"]
            + d["protein_proximity"] * W["prot"]
            + d["final_calorie_accuracy"]
            + d["final_protein_accuracy"]
        )
        constraint_penalty = (
            d["repetition_penalty"]
            + d["empty_meal_penalty"]
            + d["calorie_dense_penalty"]
            + d["cross_meal_variety"]
            + d["hard_fail_signal"]
            + d["budget_gate_signal"]
        )
        efficiency_bonus = (
            d["cost_pacing"] * W["cost"]
            + d["portion_realism"] * W["portion"]
            + d["budget_utilization"]
        )
        composition_bonus = (
            d["ingredient_diversity"] * W["div"]
            + d["food_group_coverage"] * W["grp"]
            + d["global_diversity"]
            + d["meal_balance"]
            + d["satiety_state"]
        )

        total = goal_progress + constraint_penalty + efficiency_bonus + composition_bonus

        # ---- Feedback ----
        if d["calorie_proximity"] >= 0.8:
            fb.append("Great calorie balance")
        elif d["calorie_proximity"] < 0.4:
            fb.append("Calorie level far off-target")
        if d["empty_meal_penalty"] < 0:
            fb.append("Skipped meal")
        if d["hard_fail_signal"] < 0:
            fb.append("HARD FAIL triggered  episode reward will be 0")
        if d["budget_gate_signal"] <= -2.0:
            fb.append("Budget gate blown  episode reward will be 0")
        if d["satiety_state"] < 0:
            fb.append(f"Satiety out of [40,80] window: {state.satiety:.1f}")
        if is_done:
            if d["final_calorie_accuracy"] >= 2.5:
                fb.append("Excellent daily calorie accuracy!")
            if d["global_diversity"] >= 1.5:
                fb.append("Great ingredient variety!")
            if d["meal_balance"] < 0.5:
                fb.append("Very uneven calorie distribution")

        return Reward(
            score=round(total, 4),
            breakdown=RewardBreakdown(
                goal_progress=round(goal_progress, 4),
                constraint_penalty=round(constraint_penalty, 4),
                efficiency_bonus=round(efficiency_bonus, 4),
                composition_bonus=round(composition_bonus, 4),
                details={k: round(v, 4) for k, v in d.items()},
            ),
            feedback=". ".join(fb),
        )

    # ------------------------------------------------------------------ #
    #  TIER 1  PER-MEAL SCORES                                          #
    # ------------------------------------------------------------------ #

    def _calorie_proximity(self, meal: str, actual: float, c: ConstraintConfig) -> float:
        lo = c.calorie_target * MEAL_CAL_FRACTIONS[meal][0]
        hi = c.calorie_target * MEAL_CAL_FRACTIONS[meal][1]
        return self._range_score(actual, lo, hi)

    def _protein_proximity(self, meal: str, actual: float, c: ConstraintConfig) -> float:
        if not c.meal_targets or meal not in c.meal_targets:
            return 0.0
        t = c.meal_targets[meal]
        if t.min_protein is None or t.max_protein is None:
            return 0.0
        return self._range_score(actual, t.min_protein, t.max_protein)

    def _cost_pacing(self, meal: str, actual: float, budget: float) -> float:
        MEAL_COST_FRACTIONS = {
            "breakfast": 0.20, "lunch": 0.30, "dinner": 0.35, "snack": 0.15,
        }
        expected = budget * MEAL_COST_FRACTIONS.get(meal, 0.25)
        if expected <= 0:
            return 1.0
        deviation = abs(actual - expected) / expected
        return max(0.0, 1.0 - deviation)

    def _ingredient_diversity(self, items: List[IngredientItem]) -> float:
        n = len({it.ingredient for it in items})
        return {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.8}.get(n, 1.0)

    def _food_group_coverage(self, items: List[IngredientItem]) -> float:
        groups = {FOOD_GROUPS.get(it.ingredient) for it in items} - {None}
        n = len(groups)
        return {0: 0.0, 1: 0.3, 2: 0.7}.get(n, 1.0)

    def _portion_realism(self, items: List[IngredientItem]) -> float:
        if not items:
            return 0.0
        ok = 0
        for it in items:
            grp = FOOD_GROUPS.get(it.ingredient)
            if grp and grp in REALISTIC_PORTIONS:
                lo, hi = REALISTIC_PORTIONS[grp]
                if lo <= it.quantity <= hi:
                    ok += 1
            elif 10 <= it.quantity <= 500:
                ok += 1
        return ok / len(items)

    # ------------------------------------------------------------------ #
    #  TIER 2  ANTI-GAMING PENALTIES                                     #
    # ------------------------------------------------------------------ #

    def _repetition_penalty(
        self, items: List[IngredientItem], state: NutrisyncState, current_meal: str
    ) -> float:
        prev: Set[str] = set()
        for name, built in state.meals_built.items():
            if name != current_meal and built:
                prev.update(it.ingredient for it in built)
        if not prev:
            return 0.0
        repeated = sum(1 for it in items if it.ingredient in prev)
        return max(-0.6, -0.15 * repeated)

    def _calorie_dense_penalty(self, items: List[IngredientItem], total_cal: float) -> float:
        if not items or total_cal <= 0:
            return 0.0
        dense_cal = 0.0
        for it in items:
            if it.ingredient in HIGH_DENSITY_ITEMS and it.ingredient in self.nutrition_db:
                dense_cal += self.nutrition_db[it.ingredient].calories * (it.quantity / 100.0)
        ratio = dense_cal / total_cal
        if ratio > 0.50:
            return -0.30
        elif ratio > 0.30:
            return -0.15
        return 0.0

    # ------------------------------------------------------------------ #
    #  TIER 3  EPISODE COMPLETION BONUSES                                #
    # ------------------------------------------------------------------ #

    def _final_calorie_accuracy(self, s: NutrisyncState, c: ConstraintConfig) -> float:
        if c.calorie_target <= 0:
            return 0.0
        dev = abs(1.0 - s.total_calories_consumed / c.calorie_target)
        if dev <= 0.05:  return 3.0
        if dev <= 0.10:  return 2.0
        if dev <= 0.20:  return 1.0
        return max(0.0, 1.0 - (dev - 0.20) * 5)

    def _final_protein_accuracy(self, s: NutrisyncState, c: ConstraintConfig) -> float:
        if not c.protein_target or c.protein_target <= 0:
            return 0.0
        dev = abs(1.0 - s.total_protein_consumed / c.protein_target)
        if dev <= 0.05:  return 2.0
        if dev <= 0.10:  return 1.5
        if dev <= 0.20:  return 0.8
        return max(0.0, 0.8 - (dev - 0.20) * 4)

    def _budget_utilization(self, s: NutrisyncState, c: ConstraintConfig) -> float:
        if c.budget <= 0:
            return 0.0
        u = s.total_cost_spent / c.budget
        if 0.70 <= u <= 0.95:  return 1.0
        if 0.50 <= u < 0.70:   return 0.5
        if 0.95 < u <= 1.00:   return 0.7
        return 0.2

    def _global_diversity(self, s: NutrisyncState) -> float:
        all_ing: Set[str] = set()
        for built in s.meals_built.values():
            all_ing.update(it.ingredient for it in built)
        n = len(all_ing)
        if n >= 12:  return 2.0
        if n >= 10:  return 1.5
        if n >= 7:   return 1.0
        if n >= 4:   return 0.5
        return 0.0

    def _meal_balance(self, s: NutrisyncState, c: ConstraintConfig) -> float:
        if c.calorie_target <= 0:
            return 0.0
        fracs = [0.25, 0.35, 0.30, 0.10]
        meals = ["breakfast", "lunch", "dinner", "snack"]
        deviations = []
        for i, name in enumerate(meals):
            ideal = c.calorie_target * fracs[i]
            if ideal <= 0:
                continue
            actual = sum(
                self.nutrition_db[it.ingredient].calories * (it.quantity / 100.0)
                for it in s.meals_built.get(name, [])
                if it.ingredient in self.nutrition_db
            )
            deviations.append(abs(actual - ideal) / ideal)
        if not deviations:
            return 0.0
        avg_dev = sum(deviations) / len(deviations)
        return max(0.0, 2.0 * (1.0 - avg_dev))

    def _cross_meal_variety(self, s: NutrisyncState) -> float:
        counts: Dict[str, int] = {}
        for built in s.meals_built.values():
            seen = {it.ingredient for it in built}
            for ing in seen:
                counts[ing] = counts.get(ing, 0) + 1
        penalty = 0.0
        for count in counts.values():
            if count >= 4:    penalty -= 0.5
            elif count >= 3:  penalty -= 0.3
        return max(-1.5, penalty)

    # ------------------------------------------------------------------ #
    #  HELPERS                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _range_score(actual: float, lo: float, hi: float) -> float:
        tolerance = hi - lo
        if tolerance <= 0:
            return 1.0 if actual >= lo else 0.0
        if lo <= actual <= hi:
            return 1.0
        if actual < lo:
            distance = lo - actual
        else:
            distance = actual - hi
        return max(0.0, 1.0 - distance / tolerance)
