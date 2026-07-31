"""Config contract between frontend and backend.

This module is the single source of truth for slider ranges, feasibility
rules, and preset definitions. The frontend should build its sliders from
GET /presets rather than hardcoding any of this.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

DATES_PER_YEAR = 252
T_CHOICES = (0.5, 1.0, 2.0, 5.0)

KAPPA_MIN, KAPPA_MAX = 0.5, 50.0
SIGMA_MIN = 0.5
R_MIN, R_MAX = 0.0, 0.10
Q_MAX_CAP = 1_000_000

# Baseline spot, thesis config. compute_boundary() needs a spot price to
# centre the lattice (X_0 = S_0 - alpha(0)); the request schema has no field
# for it, so it's a fixed constant rather than something exposed per-preset.
S_0 = 3.4

# K is bounded at a multiple of the preset's alpha(0) level, both here and in
# presets_payload(). Keep this as the one place that multiple is defined.
K_MAX_MULTIPLE = 2.0

# max_exercises/min_exercises are restricted to multiples of this, which
# keeps the /solve cache-key space bounded (see the ladder-ceiling check in
# _check() and _max_exercises_for_T() below) rather than letting every
# integer in [1, N] be a distinct cache entry. 21 is the coarsest step that
# lands the baseline pair (126, 105) exactly.
LADDER_STEP = 21

# Seasonal presets. `level` is alpha(0) and sets the scale for K and sigma.
# `amp` and `phase` describe the annual harmonic; service._alpha_func() turns
# these into the alpha_func the solver calls.
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

# Derived from the dicts above so there is exactly one place that lists the
# valid values. Literal[tuple(...)] is equivalent to Literal[a, b, c, ...].
AlphaPreset = Literal[tuple(ALPHA_PRESETS.keys())]
Horizon = Literal[T_CHOICES]


class SolveRequest(BaseModel):
    """A fully specified solve configuration.

    Capacity is expressed in exercise counts, not absolute volume. Because the
    arithmetic payoff is affine, S*(i, C) is invariant to a common rescaling of
    q_max and max_exercises, so the solver runs at q_max = 1 and q_max below is
    used only to label the display axis.
    """

    T: Horizon = Field(1.0, description="Contract horizon in years")
    alpha_preset: AlphaPreset = Field(DEFAULT_PRESET)

    kappa: float = Field(29.218, ge=KAPPA_MIN, le=KAPPA_MAX)
    sigma: float = Field(12.753, ge=SIGMA_MIN)
    r: float = Field(0.04, ge=R_MIN, le=R_MAX)
    K: float = Field(3.4004, gt=0.0)

    max_exercises: int = Field(
        126, ge=LADDER_STEP, multiple_of=LADDER_STEP,
        description="Number of full exercises available over the contract",
    )
    min_exercises: int = Field(
        105, ge=0, multiple_of=LADDER_STEP,
        description="Minimum number of full exercises required",
    )

    q_max: int = Field(
        10, ge=1, le=Q_MAX_CAP, description="Display scale only",
    )

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
        sigma_max = self.preset["sigma_max"]
        if self.sigma > sigma_max:
            raise ValueError(
                f"sigma {self.sigma} exceeds {sigma_max} for preset "
                f"'{self.alpha_preset}'"
            )

        k_max = K_MAX_MULTIPLE * self.preset["level"]
        if self.K > k_max:
            raise ValueError(f"K {self.K} exceeds {k_max} for this preset")

        # Check the N bound before the ladder-ceiling bound below: an
        # off-N value like max_exercises=253 at N=252 should be explained by
        # "exceeds N", not by the (unrelated) ladder ceiling.
        if self.max_exercises > self.N:
            raise ValueError(
                f"max_exercises {self.max_exercises} exceeds N = {self.N}"
            )

        ladder_ceiling = min(self.N, DATES_PER_YEAR)
        if self.max_exercises > ladder_ceiling:
            raise ValueError(
                f"max_exercises {self.max_exercises} exceeds the capacity "
                f"ladder ceiling of {ladder_ceiling} (multiples of "
                f"{LADDER_STEP} up to min(N, {DATES_PER_YEAR}))"
            )

        if self.min_exercises > self.max_exercises:
            raise ValueError("min_exercises cannot exceed max_exercises")

        return self

    def cache_key(self) -> str:
        import hashlib
        import json

        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _max_exercises_for_T(T: float) -> int:
    """Indicative work-budget bound on max_exercises for a given horizon,
    rounded down to a valid ladder rung.

    Evaluated at the calibrated kappa/sigma. At lower kappa the price lattice
    is wider and the true bound is smaller; service.check_budget() is the
    real backstop enforced on every /solve call, this is just for the slider.
    """
    from .service import WORK_BUDGET, estimate_price_nodes  # lazy: avoids import cycle

    N = int(round(T * DATES_PER_YEAR))
    probe = SolveRequest(T=T, max_exercises=LADDER_STEP, min_exercises=0)
    price_nodes = estimate_price_nodes(probe)
    max_by_budget = WORK_BUDGET / max(N * price_nodes, 1.0) - 1

    ladder_ceiling = min(N, DATES_PER_YEAR)
    bound = min(ladder_ceiling, int(max_by_budget))
    rungs = max(1, bound // LADDER_STEP)
    return rungs * LADDER_STEP


def presets_payload() -> dict[str, Any]:
    """Everything the frontend needs to build its controls."""
    from .service import WORK_BUDGET  # lazy: avoids import cycle

    return {
        "dates_per_year": DATES_PER_YEAR,
        "T_choices": list(T_CHOICES),
        "work_budget": WORK_BUDGET,
        "exercise_ladder_step": LADDER_STEP,
        "max_exercises_by_T": [
            {"T": t, "max_exercises": _max_exercises_for_T(t)} for t in T_CHOICES
        ],
        "ranges": {
            "kappa": {"min": KAPPA_MIN, "max": KAPPA_MAX, "step": 0.1},
            "sigma": {"min": SIGMA_MIN, "step": 0.1},
            "r": {"min": R_MIN, "max": R_MAX, "step": 0.01},
            "K": {"min": 0.0, "step": 0.1},
            "q_max": {"min": 1, "max": Q_MAX_CAP, "step": 1},
        },
        "alpha_presets": {
            key: {
                "label": val["label"],
                "level": val["level"],
                "sigma_max": val["sigma_max"],
                "K_max": K_MAX_MULTIPLE * val["level"],
                "illustrative": val["illustrative"],
            }
            for key, val in ALPHA_PRESETS.items()
        },
        "calibrated": SolveRequest().model_dump(),
    }
