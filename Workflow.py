import pandas
from metaflow import FlowSpec, step
import os
#os.environ["JAX_PLATFORM_NAME"] = "cpu"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax
import mlflow

from Data.Preprocessing import GetJaxArrays
from Models.LinearRegressionPipeline import RunLinearRegressionPipeline
from Models.LogisticRegressionPipeline import RunLogisticRegressionPipeline
from Models.DecisionTreePipeline import RunDecisionTreePipeline
from Models.MLPPipeline import RunMLPPipeline
from Models.AdaBoostPipeline import RunAdaBoostPipeline
from Utils.MlflowUtils import ConfigureMlflow
from Utils.SplitDataUtils import StratifiedTrainTestIndices, PrintClassDistribution

jax.config.update("jax_enable_x64", True)


class WorkFlow(FlowSpec):

    @step
    def start(self) -> None:
        self.trainFraction = 0.8
        self.next(self.load_data)

    @step
    def load_data(self):
        path = "Data/hwc.csv"
        self.dataFrame = pandas.read_csv(path)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        ConfigureMlflow()

        self.X, self.Y, self.featureNames = GetJaxArrays(self.dataFrame)
        self.featureCountBeforeIntercept = int(self.X.shape[1])
        self.sampleCount = int(self.X.shape[0])
        self.classCount = int(jax.numpy.max(self.Y)) + 1

        PrintClassDistribution("Full dataset", self.Y)

        trainIndices, testIndices = StratifiedTrainTestIndices(
            self.Y,
            trainFraction=self.trainFraction,
            seed=0,
        )
        self.splits = [(trainIndices, testIndices)]

        YTraining = self.Y[trainIndices]
        YTest = self.Y[testIndices]

        PrintClassDistribution("Training set", YTraining)
        PrintClassDistribution("Test set", YTest)

        with mlflow.start_run(run_name="feature_engineering"):
            mlflow.log_param("target", "P_HABITABLE")
            mlflow.log_param("class_count", self.classCount)
            mlflow.log_param("sample_count", self.sampleCount)
            mlflow.log_param("feature_count_before_intercept", self.featureCountBeforeIntercept)
            mlflow.log_param("train_fraction", self.trainFraction)
            mlflow.log_param("training_sample_count", int(trainIndices.shape[0]))
            mlflow.log_param("test_sample_count", int(testIndices.shape[0]))
            mlflow.log_artifact("X_features.csv")

        self.next(self.linear_regression)

    @step
    def linear_regression(self):
        ConfigureMlflow()
        self.linearResults = RunLinearRegressionPipeline(
            self.X,
            self.Y,
            self.splits,
            self.classCount,
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
        )
        self.next(self.decision_tree)

    @step
    def decision_tree(self):
        ConfigureMlflow()
        self.decisionTreeResults = RunDecisionTreePipeline(
            self.X,
            self.Y,
            self.splits,
            self.classCount,
            self.featureNames,
        )
        self.next(self.mlp)

    @step
    def mlp(self):
        ConfigureMlflow()
        self.mlpResults = RunMLPPipeline(
            self.X,
            self.Y,
            self.splits,
            self.classCount,
        )
        self.next(self.adaboost_linear)

    @step
    def adaboost_linear(self):
        ConfigureMlflow()
        self.adaboostLinearResults = RunAdaBoostPipeline(
            self.X,
            self.Y,
            self.splits,
            self.classCount,
            "linear",
        )
        self.next(self.adaboost_logistic)

    @step
    def adaboost_logistic(self):
        ConfigureMlflow()
        self.adaboostLogisticResults = RunAdaBoostPipeline(
            self.X,
            self.Y,
            self.splits,
            self.classCount,
            "logistic",
        )
        self.next(self.adaboost_decision_tree)

    @step
    def adaboost_decision_tree(self):
        ConfigureMlflow()
        self.adaboostDecisionTreeResults = RunAdaBoostPipeline(
            self.X,
            self.Y,
            self.splits,
            self.classCount,
            "decision_tree",
        )
        self.next(self.end)

    @step
    def end(self):
        print("Done")
        print("Run 'mlflow ui' in this directory to inspect the logged runs.")
