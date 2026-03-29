import pandas
from metaflow import FlowSpec, step
import jax
import mlflow

from Data.Preprocessing import GetJaxArrays
from Models.LinearRegressionPipeline import RunLinearRegressionPipeline
from Models.LogisticRegressionPipeline import RunLogisticRegressionPipeline
from Utils.MlflowUtils import ConfigureMlflow
from Utils.SplitDataUtils import StratifiedKFoldIndices, StratifiedTrainTestIndices, PrintClassDistribution

jax.config.update("jax_enable_x64", True)


class WorkFlow(FlowSpec):

    @step
    def start(self) -> None:
        self.useCrossValidation = False
        self.kFolds = 5
        self.next(self.load_data)

    @step
    def load_data(self):
        # path = kagglehub.dataset_download("chandrimad31/phl-exoplanet-catalog")
        # path = path + "/phl_exoplanet_catalog_2019.csv" # too old
        path = "Data/hwc.csv"
        self.dataFrame = pandas.read_csv(path)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        ConfigureMlflow()

        self.X, self.Y, _ = GetJaxArrays(self.dataFrame)
        self.featureCountBeforeIntercept = int(self.X.shape[1])
        self.sampleCount = int(self.X.shape[0])
        self.classCount = int(jax.numpy.max(self.Y)) + 1

        PrintClassDistribution("Full dataset", self.Y)

        if self.useCrossValidation:
            self.splits = StratifiedKFoldIndices(self.Y, k=self.kFolds, seed=0)
        else:
            trainIndices, testIndices = StratifiedTrainTestIndices(
                self.Y,
                trainFraction=0.8,
                seed=0
            )
            self.splits = [(trainIndices, testIndices)]

            XTraining = self.X[trainIndices, :]
            YTraining = self.Y[trainIndices]
            XTest = self.X[testIndices, :]
            YTest = self.Y[testIndices]

            PrintClassDistribution("Training set", YTraining)
            PrintClassDistribution("Test set", YTest)

        with mlflow.start_run(run_name="feature_engineering"):
            mlflow.log_param("target", "P_HABITABLE")
            mlflow.log_param("class_count", self.classCount)
            mlflow.log_param("sample_count", self.sampleCount)
            mlflow.log_param("feature_count_before_intercept", self.featureCountBeforeIntercept)
            mlflow.log_param("use_cross_validation", self.useCrossValidation)

            if self.useCrossValidation:
                mlflow.log_param("k_folds", self.kFolds)
            else:
                mlflow.log_param("training_sample_count", int(self.splits[0][0].shape[0]))
                mlflow.log_param("test_sample_count", int(self.splits[0][1].shape[0]))

            mlflow.log_artifact("X_features.csv")

        self.next(self.linear_regression)

    @step
    def linear_regression(self):
        ConfigureMlflow()
        self.linearResults = RunLinearRegressionPipeline(
            self.X,
            self.Y,
            self.splits,
            self.useCrossValidation,
            self.kFolds,
        )
        self.next(self.logistic_regression)

    @step
    def logistic_regression(self):
        ConfigureMlflow()
        self.logisticResults = RunLogisticRegressionPipeline(
            self.X,
            self.Y,
            self.splits,
            self.classCount,
            self.useCrossValidation,
            self.kFolds,
        )
        self.next(self.end)

    @step
    def end(self):
        print("Done")
        print("Run 'mlflow ui' in this directory to inspect the logged runs.")
