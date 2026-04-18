from __future__ import annotations

import os
import threading
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_PATH = PROJECT_ROOT / "mlruns"
EXPECTED_MODELS = [
    "linear_regression",
    "logistic_regression",
    "decision_tree",
    "mlp",
    "adaboost_linear",
    "adaboost_logistic",
    "adaboost_decision_tree",
]

app = FastAPI(title="Exoplanet Results Dashboard")
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "WebApp" / "static")), name="static")

mlflow.set_tracking_uri(str(MLRUNS_PATH.resolve().as_uri()))
_client = MlflowClient()

_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "last_return_code": None,
    "last_error": None,
    "last_stdout": "",
    "last_stderr": "",
}
_lock = threading.Lock()


class RunResponse(BaseModel):
    accepted: bool
    message: str


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _download_artifact_url(run_id: str, artifact_path: str | None) -> str | None:
    if not artifact_path:
        return None
    return f"/api/artifact/{run_id}/{artifact_path}"


def _find_latest_runs() -> dict[str, Any]:
    results: dict[str, Any] = {"models": {}, "generated_at": int(time.time())}
    experiments = _client.search_experiments()
    if not experiments:
        return results

    for experiment in experiments:
        runs = _client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=200,
        )
        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "")
            if run_name not in EXPECTED_MODELS:
                continue
            if run_name in results["models"]:
                continue

            metrics = run.data.metrics
            params = run.data.params
            artifacts = {item.path: item for item in _client.list_artifacts(run.info.run_id)}

            model_payload = {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "model_name": run_name,
                "start_time": run.info.start_time,
                "status": run.info.status,
                "metrics": {
                    "test_accuracy": _safe_float(metrics.get("test_accuracy")),
                    "macro_precision": _safe_float(metrics.get("macro_precision")),
                    "macro_recall": _safe_float(metrics.get("macro_recall")),
                    "macro_f1": _safe_float(metrics.get("macro_f1")),
                    "tree_depth": _safe_float(metrics.get("tree_depth")),
                    "leaf_count": _safe_float(metrics.get("leaf_count")),
                },
                "params": params,
                "artifacts": {
                    "confusion_plot": None,
                    "metrics_plot": None,
                    "details": None,
                },
            }

            for artifact_name in artifacts:
                lower = artifact_name.lower()
                if "confusion_matrix" in lower:
                    model_payload["artifacts"]["confusion_plot"] = _download_artifact_url(run.info.run_id, artifact_name)
                elif lower.endswith("_metrics.png"):
                    model_payload["artifacts"]["metrics_plot"] = _download_artifact_url(run.info.run_id, artifact_name)
                elif lower.endswith(".txt"):
                    model_payload["artifacts"]["details"] = _download_artifact_url(run.info.run_id, artifact_name)

            results["models"][run_name] = model_payload
            if len(results["models"]) == len(EXPECTED_MODELS):
                return results

    return results


def _run_workflow() -> None:
    command = ["python", "Main.py", "run"]
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")

    with _lock:
        _state["running"] = True
        _state["started_at"] = int(time.time())
        _state["finished_at"] = None
        _state["last_return_code"] = None
        _state["last_error"] = None
        _state["last_stdout"] = ""
        _state["last_stderr"] = ""

    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )

    with _lock:
        _state["running"] = False
        _state["finished_at"] = int(time.time())
        _state["last_return_code"] = process.returncode
        _state["last_stdout"] = process.stdout[-15000:]
        _state["last_stderr"] = process.stderr[-15000:]
        if process.returncode != 0:
            _state["last_error"] = "Workflow execution failed. Check stderr in /api/status."


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = PROJECT_ROOT / "WebApp" / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    with _lock:
        return dict(_state)


@app.post("/api/run", response_model=RunResponse)
def run_workflow() -> RunResponse:
    with _lock:
        if _state["running"]:
            return RunResponse(accepted=False, message="Workflow is already running.")
    thread = threading.Thread(target=_run_workflow, daemon=True)
    thread.start()
    return RunResponse(accepted=True, message="Workflow started.")


@app.get("/api/results/latest")
def latest_results() -> dict[str, Any]:
    return _find_latest_runs()


@app.get("/api/artifact/{run_id}/{artifact_path:path}")
def get_artifact(run_id: str, artifact_path: str):
    try:
        local_path = _client.download_artifacts(run_id, artifact_path)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(local_path)
