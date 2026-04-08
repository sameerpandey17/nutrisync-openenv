# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Production-grade HTTP client for Nutrisync RL Environment.

Provides a simple Python client interface to communicate with the
FastAPI Nutrisync server over HTTP.
"""

import logging
from typing import Any, Dict, Literal, Optional

import requests

from models import NutrisyncAction as Action, NutrisyncObservation as Observation, Reward, NutrisyncState as State

logger = logging.getLogger(__name__)


class NutrisyncClient:
    """
    HTTP client for Nutrisync RL Environment.
    
    Provides methods to interact with a running Nutrisync server.
    
    Example:
        >>> client = NutrisyncClient(base_url="http://localhost:8000")
        >>> obs = client.reset(difficulty="medium")
        >>> action = Action(items=[{"ingredient": "rice", "quantity": 100}])
        >>> obs, reward, done, info = client.step(action)
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize HTTP client.
        
        Args:
            base_url: URL of Nutrisync server (default: localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
        logger.info(f"Initialized NutrisyncClient: {self.base_url}")

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to server.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body (for POST)
        
        Returns:
            JSON response
        
        Raises:
            requests.exceptions.RequestException on error
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {method} {url}: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if server is healthy.
        
        Returns:
            True if server is responding, False otherwise
        """
        try:
            result = self._request("GET", "/health")
            return result.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def reset(
        self,
        difficulty: Literal["easy", "medium", "expert"] = "medium",
        calorie_target: float = 2000.0,
        protein_target: Optional[float] = None,
        budget: float = 200.0,
        diet_type: Literal["omnivore", "vegetarian", "vegan"] = "omnivore",
        allergies: Optional[list[str]] = None,
        seed: Optional[int] = None,
        ingredient_usage_limits: Optional[Dict[str, int]] = None,
    ) -> Observation:
        """
        Reset environment with parameters.
        
        Args:
            difficulty: EASY/MEDIUM/EXPERT
            calorie_target: Daily calorie target
            protein_target: Daily protein target
            budget: Total budget
            diet_type: omnivore/vegetarian/vegan
            allergies: List of allergenic ingredients
            seed: Random seed
            ingredient_usage_limits: Max usage per ingredient
        
        Returns:
            Initial observation
        """
        request_data = {
            "difficulty": difficulty,
            "calorie_target": calorie_target,
            "protein_target": protein_target,
            "budget": budget,
            "diet_type": diet_type,
            "allergies": allergies or [],
            "seed": seed,
            "ingredient_usage_limits": ingredient_usage_limits or {},
        }
        
        result = self._request("POST", "/reset", request_data)
        return Observation(**result)

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action object with ingredients
        
        Returns:
            (observation, reward, done, info)
        """
        request_data = {
            "items": [
                {"ingredient": item.ingredient, "quantity": item.quantity, "cooking_method": item.cooking_method}
                for item in action.items
            ]
        }
        
        result = self._request("POST", "/step", request_data)
        
        return (
            Observation(**result["observation"]),
            Reward(**result["reward"]),
            result["done"],
            result["info"],
        )

    def state(self) -> State:
        """
        Get full internal state.
        
        Returns:
            Current episode state
        """
        result = self._request("GET", "/state")
        return State(**result)

    def summary(self) -> Dict[str, Any]:
        """
        Get episode summary.
        
        Returns:
            Summary of episode
        """
        return self._request("GET", "/summary")

    def schema(self) -> Dict[str, Any]:
        """
        Get action/observation schemas.
        
        Returns:
            JSON schemas
        """
        return self._request("GET", "/schema")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_local_client() -> NutrisyncClient:
    """Create client connected to localhost server."""
    return NutrisyncClient("http://localhost:8000")


def create_client(base_url: str) -> NutrisyncClient:
    """Create client with custom server URL."""
    return NutrisyncClient(base_url)
