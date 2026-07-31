"""FastAPI entrypoint.

Note the solve route is `def`, not `async def`. FastAPI runs sync routes in a
threadpool, so a CPU-bound numpy solve will not block the event loop for every
other visitor. Declaring it `async def` would freeze the whole server for the
duration of each solve.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import ValidationError

from .schema import SolveRequest, presets_payload
from .service import solve

# Set ALLOWED_ORIGINS in the deploy environment, e.g.
#   https://<username>.github.io
_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Swing option boundary viewer", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# The boundary array dominates the payload and gzips well.
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/health")
def health():
    return {"ok": True, "version": app.version}


@app.get("/presets")
def presets():
    return presets_payload()


@app.post("/solve")
def solve_route(cfg: SolveRequest):
    try:
        return solve(cfg)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
