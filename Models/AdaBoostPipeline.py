import numpy
import mlflow

from Models.AdaBoost import BuildAdaBoostModel
from Metrics.Metrics import ConfusionMatrix, ClassificationMetrics, SaveConfusionMatrixPlot, SavePerClassMetricsPlot, PrintMetrics


def SaveAdaBoostSummary(model, outputPath, title):
    with open(outputPath, "w", encoding="utf-8") as file:
        file.write(title + "\n")
        file.write(f"estimators_used: {len(model.estimators)}\n")
        for estimatorIndex, (alpha, error) in enumerate(zip(model.estimatorWeights, model.estimatorErrors), start=1):
            file.write(f"round_{estimatorIndex}: alpha={alpha:.10f}, weighted_error={error:.10f}\n")


def RunAdaBoostPipeline(X, Y, splits, classCount, baseModelName):
    trainIndices, testIndices = splits[0]

    XTraining = numpy.asarray(X[trainIndices, :], dtype=numpy.float64)
    YTraining = numpy.asarray(Y[trainIndices]).reshape(-1).astype(int)
    XTest = numpy.asarray(X[testIndices, :], dtype=numpy.float64)
    YTest = numpy.asarray(Y[testIndices]).reshape(-1, 1).astype(int)

    model = BuildAdaBoostModel(
        baseModelName=baseModelName,
        XTraining=XTraining,
        classCount=classCount,
        randomState=1,
    )
    model.Fit(XTraining, YTraining)

    predictions = model.Predict(XTest)
    accuracy = numpy.mean(predictions == YTest)

    confusionMatrix = ConfusionMatrix(YTest, predictions, classCount)
    metrics = ClassificationMetrics(confusionMatrix)
    classLabels = [str(i) for i in range(classCount)]
    safeName = f"adaboost_{baseModelName}"

    confusionPlotPath = f"{safeName}_confusion_matrix.png"
    metricsPlotPath = f"{safeName}_metrics.png"
    summaryPath = f"{safeName}_summary.txt"

    SaveConfusionMatrixPlot(
        confusionMatrix,
        classLabels,
        confusionPlotPath,
        f"AdaBoost ({baseModelName}) Confusion Matrix",
    )
    SavePerClassMetricsPlot(
        metrics,
        classLabels,
        metricsPlotPath,
        f"AdaBoost ({baseModelName}) Precision/Recall/F1",
    )
    SaveAdaBoostSummary(
        model,
        summaryPath,
        f"AdaBoost ({baseModelName}) summary",
    )

    print(f"AdaBoost ({baseModelName}) test accuracy:", float(accuracy))
    print(f"AdaBoost ({baseModelName}) estimators used:", len(model.estimators))
    PrintMetrics(metrics)

    results = {
        "test_predictions": predictions,
        "test_accuracy": float(accuracy),
        "confusion_matrix": confusionMatrix,
        "metrics": metrics,
        "confusion_plot_path": confusionPlotPath,
        "metrics_plot_path": metricsPlotPath,
        "summary_path": summaryPath,
        "estimators_used": len(model.estimators),
        "estimator_weights": [float(x) for x in model.estimatorWeights],
        "estimator_errors": [float(x) for x in model.estimatorErrors],
        "base_model_name": str(baseModelName),
    }

    with mlflow.start_run(run_name=safeName, nested=True):
        mlflow.log_param("model", safeName)
        mlflow.log_param("class_count", classCount)
        mlflow.log_metric("test_accuracy", results["test_accuracy"])
        mlflow.log_metric("macro_precision", float(metrics["macro_precision"]))
        mlflow.log_metric("macro_recall", float(metrics["macro_recall"]))
        mlflow.log_metric("macro_f1", float(metrics["macro_f1"]))
        mlflow.log_metric("estimators_used", float(results["estimators_used"]))

        for classIndex in range(classCount):
            mlflow.log_metric(f"class_{classIndex}_precision", float(metrics["precision"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_recall", float(metrics["recall"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_f1", float(metrics["f1"][classIndex]))

        mlflow.log_artifact(confusionPlotPath)
        mlflow.log_artifact(metricsPlotPath)
        mlflow.log_artifact(summaryPath)

    return results
