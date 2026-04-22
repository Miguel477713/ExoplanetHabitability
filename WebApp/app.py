from __future__ import annotations

import os
import sys
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_PATH = PROJECT_ROOT / "mlruns"
EXPECTED_MODELS = [
    "linear_regression",
    "logistic_regression",
    "decision_tree",
    "mlp",
    #"mixture_model",
    "adaboost_linear",
    "adaboost_logistic",
    "adaboost_decision_tree",
]
DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "mlp": "MLP",
    #"mixture_model": "Mixture Model",
    "adaboost_linear": "AdaBoost Linear",
    "adaboost_logistic": "AdaBoost Logistic",
    "adaboost_decision_tree": "AdaBoost Decision Tree",
}
PREPROCESSING_GROUPS = {
    "dataset_overview": [
        "target",
        "class_count",
        "train_fraction",
        "sample_count",
        "training_sample_count",
        "test_sample_count",
    ],
    "before_preprocessing": [
        "rows_before_preprocessing",
        "columns_before_preprocessing",
        "missing_values_before_preprocessing",
        "categorical_column_count_before_encoding",
        "description_columns_requested_for_drop",
    ],
    "after_preprocessing": [
        "columns_after_column_filtering",
        "numeric_feature_count_before_imputation",
        "numeric_columns_imputed",
        "numeric_values_imputed",
        "categorical_missing_values_filled",
        "columns_after_encoding_and_imputation",
        "rows_after_preprocessing",
        "missing_values_after_preprocessing",
        "feature_count_before_intercept",
    ],
}
PREPROCESSING_LABELS = {
    "target": "Target",
    "class_count": "Class count",
    "train_fraction": "Train fraction",
    "sample_count": "Total samples",
    "training_sample_count": "Training samples",
    "test_sample_count": "Test samples",
    "rows_before_preprocessing": "Rows before preprocessing",
    "columns_before_preprocessing": "Columns before preprocessing",
    "missing_values_before_preprocessing": "Missing values before preprocessing",
    "categorical_column_count_before_encoding": "Categorical columns before encoding",
    "description_columns_requested_for_drop": "Description columns configured",
    "columns_after_column_filtering": "Columns after filtering",
    "numeric_feature_count_before_imputation": "Numeric features before imputation",
    "numeric_columns_imputed": "Numeric columns imputed",
    "numeric_values_imputed": "Missing numeric values imputed",
    "categorical_missing_values_filled": "Missing categorical values filled",
    "columns_after_encoding_and_imputation": "Columns after encoding and imputation",
    "rows_after_preprocessing": "Rows after preprocessing",
    "missing_values_after_preprocessing": "Missing values after preprocessing",
    "feature_count_before_intercept": "Feature count before intercept",
}

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


def SafeFloat(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def DownloadArtifactUrl(runId: str, artifactPath: str | None) -> str | None:
    if not artifactPath:
        return None
    return f"/api/artifact/{runId}/{artifactPath}"



def NormalizeValue(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    except Exception:
        return value



def BuildPreprocessingPayload(run: Any) -> dict[str, Any]:
    params = run.data.params
    payload = {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", "feature_engineering"),
        "start_time": run.info.start_time,
        "status": run.info.status,
        "groups": {},
    }

    for groupName, keys in PREPROCESSING_GROUPS.items():
        groupItems = []
        for key in keys:
            if key in params:
                groupItems.append({
                    "key": key,
                    "label": PREPROCESSING_LABELS.get(key, key.replace("_", " ").title()),
                    "value": NormalizeValue(params.get(key)),
                })
        payload["groups"][groupName] = groupItems

    return payload



def FindLatestFeatureEngineeringRun() -> dict[str, Any] | None:
    experiments = _client.search_experiments()
    if not experiments:
        return None

    for experiment in experiments:
        runs = _client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=100,
        )
        for run in runs:
            runName = run.data.tags.get("mlflow.runName", "")
            if runName == "feature_engineering":
                return BuildPreprocessingPayload(run)
    return None



def FindLatestRuns() -> dict[str, Any]:
    results: dict[str, Any] = {
        "models": [],
        "preprocessing": FindLatestFeatureEngineeringRun(),
        "generated_at": int(time.time()),
    }
    experiments = _client.search_experiments()
    if not experiments:
        return results

    latestByName: dict[str, Any] = {}

    for experiment in experiments:
        runs = _client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=300,
        )
        for run in runs:
            runName = run.data.tags.get("mlflow.runName", "")
            if runName not in EXPECTED_MODELS:
                continue
            if runName in latestByName:
                continue

            metrics = run.data.metrics
            params = run.data.params
            artifacts = {item.path: item for item in _client.list_artifacts(run.info.run_id)}

            modelPayload = {
                "run_id": run.info.run_id,
                "run_name": runName,
                "display_name": DISPLAY_NAMES.get(runName, runName.replace("_", " ").title()),
                "start_time": run.info.start_time,
                "status": run.info.status,
                "metrics": {
                    "accuracy": SafeFloat(metrics.get("test_accuracy")),
                    "precision": SafeFloat(metrics.get("macro_precision")),
                    "recall": SafeFloat(metrics.get("macro_recall")),
                    "f1_score": SafeFloat(metrics.get("macro_f1")),
                    "tree_depth": SafeFloat(metrics.get("tree_depth")),
                    "leaf_count": SafeFloat(metrics.get("leaf_count")),
                    "estimators_used": SafeFloat(metrics.get("estimators_used")),
                },
                "params": params,
                "artifacts": {
                    "confusion_plot": None,
                    "metrics_plot": None,
                    "details": None,
                },
            }

            for artifactName in artifacts:
                lower = artifactName.lower()
                if "confusion_matrix" in lower:
                    modelPayload["artifacts"]["confusion_plot"] = DownloadArtifactUrl(run.info.run_id, artifactName)
                elif lower.endswith("_metrics.png"):
                    modelPayload["artifacts"]["metrics_plot"] = DownloadArtifactUrl(run.info.run_id, artifactName)
                elif lower.endswith(".txt"):
                    modelPayload["artifacts"]["details"] = DownloadArtifactUrl(run.info.run_id, artifactName)

            latestByName[runName] = modelPayload
            if len(latestByName) == len(EXPECTED_MODELS):
                break

    for modelName in EXPECTED_MODELS:
        if modelName in latestByName:
            results["models"].append(latestByName[modelName])

    return results


def RunWorkflowProcess() -> None:
    command = [sys.executable, os.path.join(BASE_DIR, "Main.py"), "run"]
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
        else:
            _state["last_error"] = None


@app.get("/", response_class=HTMLResponse)
def Index() -> str:
    htmlPath = PROJECT_ROOT / "WebApp" / "templates" / "index.html"
    return htmlPath.read_text(encoding="utf-8")


@app.get("/api/status")
def GetStatus() -> dict[str, Any]:
    with _lock:
        return dict(_state)


@app.post("/api/run", response_model=RunResponse)
def RunWorkflow() -> RunResponse:
    with _lock:
        if _state["running"]:
            return RunResponse(accepted=False, message="Workflow is already running.")
    thread = threading.Thread(target=RunWorkflowProcess, daemon=True)
    thread.start()
    return RunResponse(accepted=True, message="Workflow started.")


@app.get("/api/results/latest")
def LatestResults() -> dict[str, Any]:
    return FindLatestRuns()


@app.get("/api/artifact/{runId}/{artifactPath:path}")
def GetArtifact(runId: str, artifactPath: str):
    try:
        localPath = _client.download_artifacts(runId, artifactPath)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(localPath)
