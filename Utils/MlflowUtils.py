from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = PROJECT_ROOT.joinpath("mlruns").resolve().as_uri()

def ConfigureMlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("ExoplanetsHabitability")