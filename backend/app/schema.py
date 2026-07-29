"""Config contract between frontend and backend.

This module is the single source of truth for slider ranges, feasibility
rules, and preset definitions. The frontend should build its sliders from
GET /presets rather than hardcoding any of this.
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

DATES_PER_YEAR = 252
T_CHOICES = (0.5, 1.0, 2.0, 5.0)

N_MAX_CAP = 150          # bounds the volume dimension regardless of contract size
KAPPA_MIN, KAPPA_MAX = 0.5, 50.0
SIGMA_MIN = 0.5
R_MIN, R_MAX = 0.0, 0.10

# Seasonal presets. `level` is alpha(0) and sets the scale for K and sigma.
# `amp` and `phase` describe the annual harmonic used by the stub; replace
# with the real calibrated alpha coefficients when wiring the solver.
ALPHA_PRESETS: dict[str, dict[str, Any]] = {
    "henry_hub": {
        "label": "Natural gas (Henry Hub, calibrated)",
        "level": 3.4004, "amp": 0.18, "phase": 0.0,
        "sigma_max": 30.0, "illustrative": False,
    },
    "henry_hub_hi": {
        "label": "Natural gas, amplified seasonality",
        "level": 3.4004, "amp": 0.30, "phase": 0.0,
        "sigma_max": 30.0, "illustrative": False,
    },
    "henry_hub_lo": {
        "label": "Natural gas, damped seasonality",
        "level": 3.4004, "amp": 0.09, "phase": 0.0,
        "sigma_max": 30.0, "illustrative": False,
    },
    "wti": {
        "label": "Crude oil (WTI, illustrative)",
        "level": 70.0, "amp": 0.02, "phase": 0.0,
        "sigma_max": 400.0, "illustrative": True,
    },
    "power": {
        "label": "Electricity (illustrative only, OU is misspecified)",
        "level": 50.0, "amp": 0.35, "phase": 0.25,
        "sigma_max": 600.0, "illustrative": True,
    },
}

DEFAULT_PRESET = "henry_hub"


class SolveRequest(BaseModel):
    """A fully specified solve configuration.

    Capacity is expressed in exercise counts, not absolute volume. Because the
    arithmetic payoff is affine, S*(i, C) is invariant to a common rescaling of
    q_max and C_max, so the solver runs at q_max = 1 and q_max below is used
    only to label the display axis.
    """

    T: float = Field(1.0, description="Contract horizon in years")
    alpha_preset: str = Field(DEFAULT_PRESET)

    kappa: float = Field(29.218, ge=KAPPA_MIN, le=KAPPA_MAX)
    sigma: float = Field(12.753, ge=SIGMA_MIN)
    r: float = Field(0.04, ge=R_MIN, le=R_MAX)
    K: float = Field(3.4004, gt=0.0)

    n_max: int = Field(126, ge=1, le=N_MAX_CAP, description="C_max / q_max")
    n_min: int = Field(105, ge=0, description="C_min / q_max")

    q_max: float = Field(10.0, gt=0.0, description="Display scale only")

    @property
    def N(self) -> int:
        return int(round(self.T * DATES_PER_YEAR))

    @property
    def preset(self) -> dict[str, Any]:
        return ALPHA_PRESETS[self.alpha_preset]

    @property
    def moneyness(self) -> float:
        return self.K / self.preset["level"]

    @property
    def stationary_sd(self) -> float:
        """sigma / sqrt(2 kappa): the scale the boundary actually responds to."""
        return self.sigma / (2.0 * self.kappa) ** 0.5

    @model_validator(mode="after")
    def _check(self) -> "SolveRequest":
        if self.alpha_preset not in ALPHA_PRESETS:
            raise ValueError(f"unknown alpha_preset '{self.alpha_preset}'")

        if self.T not in T_CHOICES:
            raise ValueError(f"T must be one of {T_CHOICES}")

        sigma_max = self.preset["sigma_max"]
        if self.sigma > sigma_max:
            raise ValueError(
                f"sigma {self.sigma} exceeds {sigma_max} for preset "
                f"'{self.alpha_preset}'"
            )

        k_max = 2.0 * self.preset["level"]
        if self.K > k_max:
            raise ValueError(f"K {self.K} exceeds {k_max} for this preset")

        if self.n_min > self.n_max:
            raise ValueError("n_min cannot exceed n_max")

        if self.n_max > self.N:
            raise ValueError(f"n_max {self.n_max} exceeds N = {self.N}")

        if self.n_min > self.N:
            raise ValueError(
                f"n_min {self.n_min} is unreachable in N = {self.N} dates"
            )

        return self

    def cache_key(self) -> str:
        import hashlib
        import json

        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def presets_payload() -> dict[str, Any]:
    """Everything the frontend needs to build its controls."""
    return {
        "dates_per_year": DATES_PER_YEAR,
        "T_choices": list(T_CHOICES),
        "n_max_cap": N_MAX_CAP,
        "ranges": {
            "kappa": {"min": KAPPA_MIN, "max": KAPPA_MAX, "step": 0.1},
            "sigma": {"min": SIGMA_MIN, "step": 0.1},
            "r": {"min": R_MIN, "max": R_MAX, "step": 0.01},
            "K": {"min": 0.0, "step": 0.1},
        },
        "alpha_presets": {
            key: {
                "label": val["label"],
                "level": val["level"],
                "sigma_max": val["sigma_max"],
                "K_max": 2.0 * val["level"],
                "illustrative": val["illustrative"],
            }
            for key, val in ALPHA_PRESETS.items()
        },
        "calibrated": SolveRequest().model_dump(),
    }
