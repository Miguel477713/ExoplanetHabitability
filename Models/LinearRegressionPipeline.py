import numpy
import jax
import mlflow

from Models.LinearRegression import LinearRegressionCoefficients, EstimateWithLinearRegression
from Utils.JaxUtils import AddIntercept
from Metrics.Metrics import ConfusionMatrix, ClassificationMetrics, SaveConfusionMatrixPlot, SavePerClassMetricsPlot, PrintMetrics


def RunLinearRegressionPipeline(X, Y, splits, classCount):
    trainIndices, testIndices = splits[0]

    XTraining = X[trainIndices, :]
    YTraining = Y[trainIndices]
    XTest = X[testIndices, :]
    YTest = Y[testIndices]

    XTraining = AddIntercept(XTraining)
    XTest = AddIntercept(XTest)

    linearRegressionB = LinearRegressionCoefficients(XTraining, YTraining)

    linearTestPredictions = []
    for rowIndex in range(XTest.shape[0]):
        prediction = EstimateWithLinearRegression(
            linearRegressionB,
            XTest[rowIndex].reshape(-1, 1),
            isClassification=True,
        )
        linearTestPredictions.append(prediction)

    linearTestPredictions = jax.numpy.array(linearTestPredictions).reshape(-1, 1)
    linearTestAccuracy = jax.numpy.mean(linearTestPredictions == YTest)
    linearTestMSE = jax.numpy.mean((XTest @ linearRegressionB - YTest) ** 2)

    confusionMatrix = ConfusionMatrix(YTest, linearTestPredictions, classCount)
    metrics = ClassificationMetrics(confusionMatrix)
    classLabels = [str(i) for i in range(classCount)]

    confusionPlotPath = "linear_confusion_matrix.png"
    metricsPlotPath = "linear_metrics.png"
    coefficientsPath = "linear_coefficients.txt"

    SaveConfusionMatrixPlot(
        confusionMatrix,
        classLabels,
        confusionPlotPath,
        "Linear Regression Confusion Matrix",
    )
    SavePerClassMetricsPlot(
        metrics,
        classLabels,
        metricsPlotPath,
        "Linear Regression Precision/Recall/F1",
    )
    numpy.savetxt(coefficientsPath, numpy.asarray(linearRegressionB), fmt="%.10f")

    print("Linear regression test accuracy:", float(linearTestAccuracy))
    print("Linear regression test mse:", float(linearTestMSE))
    PrintMetrics(metrics)

    results = {
        "coefficients": linearRegressionB,
        "test_predictions": linearTestPredictions,
        "test_accuracy": float(linearTestAccuracy),
        "test_mse": float(linearTestMSE),
        "confusion_matrix": confusionMatrix,
        "metrics": metrics,
        "confusion_plot_path": confusionPlotPath,
        "metrics_plot_path": metricsPlotPath,
        "coefficients_path": coefficientsPath,
    }

    with mlflow.start_run(run_name="linear_regression", nested=True):
        mlflow.log_param("model", "linear_regression_classifier")
        mlflow.log_param("feature_count_with_intercept", int(XTraining.shape[1]))
        mlflow.log_metric("test_accuracy", results["test_accuracy"])
        mlflow.log_metric("test_mse", results["test_mse"])
        mlflow.log_metric("macro_precision", float(metrics["macro_precision"]))
        mlflow.log_metric("macro_recall", float(metrics["macro_recall"]))
        mlflow.log_metric("macro_f1", float(metrics["macro_f1"]))

        for classIndex in range(classCount):
            mlflow.log_metric(f"class_{classIndex}_precision", float(metrics["precision"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_recall", float(metrics["recall"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_f1", float(metrics["f1"][classIndex]))

        mlflow.log_artifact(coefficientsPath)
        mlflow.log_artifact(confusionPlotPath)
        mlflow.log_artifact(metricsPlotPath)

    return results
