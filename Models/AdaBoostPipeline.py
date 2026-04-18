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


def RunAdaBoostPipeline(X, Y, splits, classCount, useCrossValidation, kFolds, baseModelName):
    results = {
        "fold_accuracies": [],
        "fold_macro_f1": [],
        "fold_macro_precision": [],
        "fold_macro_recall": [],
        "mean_accuracy": None,
        "mean_macro_f1": None,
        "mean_macro_precision": None,
        "mean_macro_recall": None,
        "test_accuracy": None,
        "metrics": {
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
        },
    }

    classLabels = [str(i) for i in range(classCount)]
    safeName = f"adaboost_{baseModelName}"

    for splitNumber, (trainIndices, testIndices) in enumerate(splits, start=1):
        XTraining = numpy.asarray(X[trainIndices, :], dtype=numpy.float64)
        YTraining = numpy.asarray(Y[trainIndices]).reshape(-1).astype(int)
        XTest = numpy.asarray(X[testIndices, :], dtype=numpy.float64)
        YTest = numpy.asarray(Y[testIndices]).reshape(-1, 1).astype(int)

        model = BuildAdaBoostModel(
            baseModelName=baseModelName,
            XTraining=XTraining,
            classCount=classCount,
            randomState=splitNumber,
        )
        model.Fit(XTraining, YTraining)

        predictions = model.Predict(XTest)
        accuracy = numpy.mean(predictions == YTest)

        confusionMatrix = ConfusionMatrix(YTest, predictions, classCount)
        metrics = ClassificationMetrics(confusionMatrix)

        results["fold_accuracies"].append(float(accuracy))
        results["fold_macro_f1"].append(float(metrics["macro_f1"]))
        results["fold_macro_precision"].append(float(metrics["macro_precision"]))
        results["fold_macro_recall"].append(float(metrics["macro_recall"]))

        confusionPlotPath = f"{safeName}_confusion_matrix_split_{splitNumber}.png"
        metricsPlotPath = f"{safeName}_metrics_split_{splitNumber}.png"
        summaryPath = f"{safeName}_summary_split_{splitNumber}.txt"

        SaveConfusionMatrixPlot(
            confusionMatrix,
            classLabels,
            confusionPlotPath,
            f"AdaBoost ({baseModelName}) Confusion Matrix - Split {splitNumber}",
        )
        SavePerClassMetricsPlot(
            metrics,
            classLabels,
            metricsPlotPath,
            f"AdaBoost ({baseModelName}) Precision/Recall/F1 - Split {splitNumber}",
        )
        SaveAdaBoostSummary(
            model,
            summaryPath,
            f"AdaBoost ({baseModelName}) summary - split {splitNumber}",
        )

        print(f"AdaBoost ({baseModelName}) split {splitNumber} accuracy:", float(accuracy))
        print(f"AdaBoost ({baseModelName}) split {splitNumber} estimators used:", len(model.estimators))
        PrintMetrics(metrics)

        if not useCrossValidation and splitNumber == 1:
            results["test_predictions"] = predictions
            results["test_accuracy"] = float(accuracy)
            results["confusion_matrix"] = confusionMatrix
            results["metrics"] = metrics
            results["confusion_plot_path"] = confusionPlotPath
            results["metrics_plot_path"] = metricsPlotPath
            results["summary_path"] = summaryPath
            results["estimators_used"] = len(model.estimators)
            results["estimator_weights"] = [float(x) for x in model.estimatorWeights]
            results["estimator_errors"] = [float(x) for x in model.estimatorErrors]
            results["base_model_name"] = str(baseModelName)

    results["mean_accuracy"] = float(sum(results["fold_accuracies"]) / len(results["fold_accuracies"]))
    results["mean_macro_f1"] = float(sum(results["fold_macro_f1"]) / len(results["fold_macro_f1"]))
    results["mean_macro_precision"] = float(sum(results["fold_macro_precision"]) / len(results["fold_macro_precision"]))
    results["mean_macro_recall"] = float(sum(results["fold_macro_recall"]) / len(results["fold_macro_recall"]))

    runName = f"{safeName}_cv" if useCrossValidation else safeName

    with mlflow.start_run(run_name=runName, nested=True):
        mlflow.log_param("model", safeName)
        mlflow.log_param("class_count", classCount)

        if useCrossValidation:
            mlflow.log_param("k_folds", kFolds)
            mlflow.log_metric("mean_test_accuracy", results["mean_accuracy"])
            mlflow.log_metric("mean_macro_f1", results["mean_macro_f1"])
            mlflow.log_metric("mean_macro_precision", results["mean_macro_precision"])
            mlflow.log_metric("mean_macro_recall", results["mean_macro_recall"])

            for splitNumber, value in enumerate(results["fold_accuracies"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_accuracy", value)
            for splitNumber, value in enumerate(results["fold_macro_f1"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_macro_f1", value)
            for splitNumber, value in enumerate(results["fold_macro_precision"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_macro_precision", value)
            for splitNumber, value in enumerate(results["fold_macro_recall"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_macro_recall", value)

            for splitNumber in range(1, len(splits) + 1):
                mlflow.log_artifact(f"{safeName}_confusion_matrix_split_{splitNumber}.png")
                mlflow.log_artifact(f"{safeName}_metrics_split_{splitNumber}.png")
                mlflow.log_artifact(f"{safeName}_summary_split_{splitNumber}.txt")
        else:
            mlflow.log_metric("test_accuracy", float(results["test_accuracy"]))
            mlflow.log_metric("macro_precision", float(results["metrics"]["macro_precision"]))
            mlflow.log_metric("macro_recall", float(results["metrics"]["macro_recall"]))
            mlflow.log_metric("macro_f1", float(results["metrics"]["macro_f1"]))
            mlflow.log_metric("estimators_used", float(results["estimators_used"]))

            for classIndex in range(classCount):
                mlflow.log_metric(f"class_{classIndex}_precision", float(results["metrics"]["precision"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_recall", float(results["metrics"]["recall"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_f1", float(results["metrics"]["f1"][classIndex]))

            mlflow.log_artifact(results["confusion_plot_path"])
            mlflow.log_artifact(results["metrics_plot_path"])
            mlflow.log_artifact(results["summary_path"])

    if useCrossValidation:
        print(f"AdaBoost ({baseModelName}) mean CV accuracy:", results["mean_accuracy"])
        print(f"AdaBoost ({baseModelName}) mean CV macro F1:", results["mean_macro_f1"])
        print(f"AdaBoost ({baseModelName}) mean CV macro precision:", results["mean_macro_precision"])
        print(f"AdaBoost ({baseModelName}) mean CV macro recall:", results["mean_macro_recall"])
    else:
        print(f"AdaBoost ({baseModelName}) test accuracy:", float(results["test_accuracy"]))
        print(f"AdaBoost ({baseModelName}) test macro F1:", float(results["metrics"]["macro_f1"]))
        print(f"AdaBoost ({baseModelName}) test macro precision:", float(results["metrics"]["macro_precision"]))
        print(f"AdaBoost ({baseModelName}) test macro recall:", float(results["metrics"]["macro_recall"]))

    return results