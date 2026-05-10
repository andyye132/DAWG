"""DAWG playground — minimal MolmoWeb runner with a web UI.

Run alongside MolmoWeb's model server (default port 8001).

Required env vars (defaults in parens):
    DAWG_USER      ('dawg')      HTTP Basic Auth username
    DAWG_PASS      ('changeme')  HTTP Basic Auth password — CHANGE THIS
    MOLMOWEB_URL   ('http://127.0.0.1:8001')  model server endpoint

Launch:
    uvicorn dawg.playground.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import secrets
import sys
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

# Make MolmoWeb's `inference.client` importable.
MOLMOWEB_PATH = "/gscratch/raivn/andy132/dawg/external/molmoweb"
if MOLMOWEB_PATH not in sys.path:
    sys.path.insert(0, MOLMOWEB_PATH)

from inference.client import MolmoWeb  # noqa: E402

# ---------- Config ----------

USERNAME = os.environ.get("DAWG_USER", "dawg")
PASSWORD = os.environ.get("DAWG_PASS", "changeme")
MOLMOWEB_URL = os.environ.get("MOLMOWEB_URL", "http://127.0.0.1:8001")

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
TRAJ_DIR = Path("/gscratch/raivn/andy132/dawg/results/playground/trajectories")
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Auth ----------

security = HTTPBasic(realm="DAWG Playground")


def require_auth(creds: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    ok_user = secrets.compare_digest(creds.username.encode(), USERNAME.encode())
    ok_pass = secrets.compare_digest(creds.password.encode(), PASSWORD.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return creds.username


# ---------- App ----------

app = FastAPI(title="DAWG Playground")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Reused across requests. MolmoWeb client creates a fresh Playwright env per
# `.run()`; the model itself stays warm in the external model server.
client = MolmoWeb(endpoint=MOLMOWEB_URL, local=True, headless=True)

# Process-local map from run_id → saved trajectory HTML path. Reloaded from
# disk on demand if the process restarted.
RECENT: dict[str, Path] = {}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, _: str = Depends(require_auth)):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "default_max_steps": 10},
    )


@app.post("/run", response_class=HTMLResponse)
def run(
    request: Request,
    url: Annotated[str, Form()],
    query: Annotated[str, Form()],
    max_steps: Annotated[int, Form()] = 10,
    _: str = Depends(require_auth),
):
    full_query = f"Navigate to {url}. Then: {query}"
    try:
        traj = client.run(query=full_query, max_steps=max_steps)
    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "url": url,
                "query": query,
                "error": f"{type(e).__name__}: {e}",
                "run_id": None,
                "n_steps": 0,
            },
            status_code=500,
        )

    run_id = uuid.uuid4().hex[:10]
    html_path = TRAJ_DIR / f"{run_id}.html"
    traj.save_html(output_path=str(html_path), query=full_query)
    RECENT[run_id] = html_path

    n_steps = len(getattr(traj, "steps", []) or [])
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "url": url,
            "query": query,
            "run_id": run_id,
            "n_steps": n_steps,
            "error": None,
        },
    )


@app.get("/trajectory/{run_id}")
def trajectory(run_id: str, _: str = Depends(require_auth)):
    path = RECENT.get(run_id)
    if path is None or not path.exists():
        candidate = TRAJ_DIR / f"{run_id}.html"
        if not candidate.exists():
            raise HTTPException(404, "trajectory not found")
        RECENT[run_id] = candidate
        path = candidate
    return FileResponse(path, media_type="text/html")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "molmoweb_url": MOLMOWEB_URL}
