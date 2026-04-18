import numpy
import jax
import mlflow

from Models.LogisticRegression import LogisticRegressionCoefficients, EstimateWithLogisticRegression
from Utils.JaxUtils import AddIntercept
from Metrics.Metrics import ConfusionMatrix, ClassificationMetrics, SaveConfusionMatrixPlot, SavePerClassMetricsPlot, PrintMetrics


def RunLogisticRegressionPipeline(X, Y, splits, classCount):
    learningRate = 0.01
    iterations = 2000

    trainIndices, testIndices = splits[0]

    XTraining = X[trainIndices, :]
    YTraining = Y[trainIndices]
    XTest = X[testIndices, :]
    YTest = Y[testIndices]

    XTraining = AddIntercept(XTraining)
    XTest = AddIntercept(XTest)

    logisticRegressionB = LogisticRegressionCoefficients(
        XTraining,
        YTraining,
        classCount=classCount,
        learningRate=learningRate,
        iterations=iterations,
    )

    logisticTestPredictions = []
    for rowIndex in range(XTest.shape[0]):
        prediction = EstimateWithLogisticRegression(
            logisticRegressionB,
            XTest[rowIndex].reshape(-1, 1),
        )
        logisticTestPredictions.append(prediction)

    logisticTestPredictions = jax.numpy.array(logisticTestPredictions).reshape(-1, 1)
    logisticTestAccuracy = jax.numpy.mean(logisticTestPredictions == YTest)

    confusionMatrix = ConfusionMatrix(YTest, logisticTestPredictions, classCount)
    metrics = ClassificationMetrics(confusionMatrix)
    classLabels = [str(i) for i in range(classCount)]

    confusionPlotPath = "logistic_confusion_matrix.png"
    metricsPlotPath = "logistic_metrics.png"
    coefficientsPath = "logistic_coefficients.txt"

    SaveConfusionMatrixPlot(
        confusionMatrix,
        classLabels,
        confusionPlotPath,
        "Logistic Regression Confusion Matrix",
    )
    SavePerClassMetricsPlot(
        metrics,
        classLabels,
        metricsPlotPath,
        "Logistic Regression Precision/Recall/F1",
    )
    numpy.savetxt(coefficientsPath, numpy.asarray(logisticRegressionB), fmt="%.10f")

    print("Logistic regression test accuracy:", float(logisticTestAccuracy))
    PrintMetrics(metrics)

    results = {
        "coefficients": logisticRegressionB,
        "test_predictions": logisticTestPredictions,
        "test_accuracy": float(logisticTestAccuracy),
        "confusion_matrix": confusionMatrix,
        "metrics": metrics,
        "confusion_plot_path": confusionPlotPath,
        "metrics_plot_path": metricsPlotPath,
        "coefficients_path": coefficientsPath,
    }

    with mlflow.start_run(run_name="logistic_regression", nested=True):
        mlflow.log_param("model", "logistic_regression_classifier")
        mlflow.log_param("learning_rate", learningRate)
        mlflow.log_param("iterations", iterations)
        mlflow.log_metric("test_accuracy", results["test_accuracy"])
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
