import os
import mlflow

TRACKING_URI = "file:///mnt/d/Users/migue/Escritorio/Computer science masters/Machine learning/Final project/mlruns"

def ConfigureMlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("ExoplanetsHabitability")
