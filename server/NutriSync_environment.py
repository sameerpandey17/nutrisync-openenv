# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Nutrisync OpenEnv Interface  V2.

Provides the OpenEnv `Environment` adapter encapsulating the core V2 `NutrisyncEnv`.
Exposes 3 fixed-configuration tasks ('easy', 'medium', 'hard') with V2 graders
and surfaces all new observation fields (satiety, cluster counts, temporal constraints,
seasonal availability, daily micronutrient totals, budget gate, hard_fail).
"""

import logging
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)

from server.environment import NutrisyncEnv
from server.tasks import TASKS
try:
    from models import NutrisyncAction, NutrisyncObservation, NutrisyncState
except ImportError:
    from NutriSync.models import NutrisyncAction, NutrisyncObservation, NutrisyncState

logger = logging.getLogger(__name__)


class NutrisyncEnvironment(Environment[NutrisyncAction, NutrisyncObservation, NutrisyncState]):
    """
    NutriSync OpenEnv Wrapper  V2.

    Provides the standard OpenEnv step/reset/state interface wrapping
    the core NutrisyncEnv V2 engine. Includes support for "easy", "medium",
    and "hard" tasks with multiplicative reward pipeline.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core_env: Optional[NutrisyncEnv] = None
        self._current_task_id: str = "easy"
        self._episode_id: str = str(uuid4())

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> NutrisyncObservation:
        """
        Reset the environment.

        Args:
            seed: Optional seed for reproducibility
            episode_id: Optional string to identify this run
            task_id (kwargs): 'easy', 'medium', or 'hard'
        """
        self._episode_id = episode_id or str(uuid4())
        self._current_task_id = kwargs.get("task_id", "easy")

        if self._current_task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id: '{self._current_task_id}'. "
                f"Available: {list(TASKS.keys())}"
            )

        task_config = TASKS[self._current_task_id].copy()
        task_config.pop("grader", None)   # grader is not a NutrisyncEnv arg

        self.core_env = NutrisyncEnv(**task_config, seed=seed)
        core_obs = self.core_env.reset()
        core_state = self.core_env.get_state()

        logger.info(
            f"V2 task '{self._current_task_id}' started (Episode: {self._episode_id})"
        )

        return self._build_observation(core_obs, core_state, final_score=0.0)

    def step(
        self,
        action: NutrisyncAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> NutrisyncObservation:
        """Execute a step in the environment."""
        if not self.core_env:
            raise RuntimeError("Environment not reset. Call reset() first.")

        core_obs, reward, done, info = self.core_env.step(action)
        core_state = self.core_env.get_state()

        # At episode end: call V2 grader for final normalized score
        final_score = reward.score
        if done:
            grader = TASKS[self._current_task_id]["grader"]
            final_grade = grader(core_state)
            info["final_grade"] = final_grade
            # Overwrite with normalized [0, 1] grade for OpenEnv compatibility
            final_score = final_grade

        return self._build_observation(core_obs, core_state, final_score=final_score, info=info)

    def _build_observation(
        self,
        core_obs: NutrisyncObservation,
        core_state: NutrisyncState,
        final_score: float = 0.0,
        info: Optional[dict] = None,
    ) -> NutrisyncObservation:
        """Build enriched V2 observation with all new fields."""
        info = info or {}
        done = core_state.done

        return NutrisyncObservation(
            # Core fields
            current_meal=core_state.current_meal if not done else "done",
            calories_left=max(0.0, core_state.calories_left),
            protein_left=max(0.0, core_state.protein_left),
            budget_left=max(0.0, core_state.budget_left),
            meals_built={
                m: [
                    {
                        "ingredient": it.ingredient,
                        "quantity": it.quantity,
                        "cooking_method": it.cooking_method,
                    }
                    for it in items
                ]
                for m, items in core_state.meals_built.items()
            },
            ingredient_usage_count=core_state.ingredient_usage_count,
            allowed_ingredients=core_state.allowed_ingredients,
            constraints={
                "difficulty":      core_state.constraints.difficulty,
                "diet_type":       core_state.constraints.diet_type,
                "allergies":       core_state.constraints.allergies,
                "calorie_target":  core_state.constraints.calorie_target,
                "protein_target":  core_state.constraints.protein_target,
                "budget":          core_state.constraints.budget,
            },
            # V2: Satiety and temporal coupling
            satiety=core_state.satiety,
            cumulative_protein_g=core_state.cumulative_protein_g,
            protein_pace_deficit=core_obs.protein_pace_deficit,
            # V2: Filtered action space
            available_ingredients=core_obs.available_ingredients,
            unavailable_ingredients=sorted(core_state.unavailable_ingredients),
            # V2: Micronutrient totals
            daily_totals=dict(core_state.daily_totals),
            # V2: Cluster tracking
            cluster_usage_counts=dict(core_state.cluster_usage_counts),
            # V2: State flags
            action_space_constrained=core_state.action_space_constrained,
            budget_gate=core_state.budget_gate,
            hard_fail=core_state.hard_fail,
            # OpenEnv standard
            reward=final_score,
            done=done,
            metadata={
                "step": core_state.step_count,
                "task_id": self._current_task_id,
                "info": info,
                "feedback": (
                    core_state.reward_history[-1].feedback
                    if core_state.reward_history else ""
                ),
                "availability_violations": core_state.availability_violations,
            },
        )

    @property
    def state(self) -> NutrisyncState:
        """Get the current environment state."""
        if not self.core_env:
            raise RuntimeError("Environment not reset.")
        return self.core_env.get_state()
