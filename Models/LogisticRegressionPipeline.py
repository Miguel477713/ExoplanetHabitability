import numpy
import jax
import mlflow

from Models.LogisticRegression import LogisticRegressionCoefficients, EstimateWithLogisticRegression
from Utils.JaxUtils import AddIntercept
from Metrics.Metrics import ConfusionMatrix, ClassificationMetrics, SaveConfusionMatrixPlot, SavePerClassMetricsPlot, PrintMetrics

def RunLogisticRegressionPipeline(X, Y, splits, classCount, useCrossValidation, kFolds):
    learningRate = 0.01
    iterations = 2000

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

    for splitNumber, (trainIndices, testIndices) in enumerate(splits, start=1):
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

        results["fold_accuracies"].append(float(logisticTestAccuracy))
        results["fold_macro_f1"].append(float(metrics["macro_f1"]))
        results["fold_macro_precision"].append(float(metrics["macro_precision"]))
        results["fold_macro_recall"].append(float(metrics["macro_recall"]))

        confusionPlotPath = f"logistic_confusion_matrix_split_{splitNumber}.png"
        metricsPlotPath = f"logistic_metrics_split_{splitNumber}.png"

        SaveConfusionMatrixPlot(
            confusionMatrix,
            classLabels,
            confusionPlotPath,
            f"Logistic Regression Confusion Matrix - Split {splitNumber}",
        )
        SavePerClassMetricsPlot(
            metrics,
            classLabels,
            metricsPlotPath,
            f"Logistic Regression Precision/Recall/F1 - Split {splitNumber}",
        )

        print(f"Logistic regression split {splitNumber} accuracy:", float(logisticTestAccuracy))
        PrintMetrics(metrics)

        if not useCrossValidation and splitNumber == 1:
            results["coefficients"] = logisticRegressionB
            results["test_predictions"] = logisticTestPredictions
            results["test_accuracy"] = float(logisticTestAccuracy)
            results["confusion_matrix"] = confusionMatrix
            results["metrics"] = metrics
            results["confusion_plot_path"] = confusionPlotPath
            results["metrics_plot_path"] = metricsPlotPath
            numpy.savetxt("logistic_coefficients.txt", numpy.asarray(logisticRegressionB), fmt="%.10f")

    results["mean_accuracy"] = float(sum(results["fold_accuracies"]) / len(results["fold_accuracies"]))
    results["mean_macro_f1"] = float(sum(results["fold_macro_f1"]) / len(results["fold_macro_f1"]))
    results["mean_macro_precision"] = float(sum(results["fold_macro_precision"]) / len(results["fold_macro_precision"]))
    results["mean_macro_recall"] = float(sum(results["fold_macro_recall"]) / len(results["fold_macro_recall"]))

    runName = "logistic_regression_cv" if useCrossValidation else "logistic_regression"

    with mlflow.start_run(run_name=runName):
        mlflow.log_param("model", "multiclass_logistic_regression")
        mlflow.log_param("class_count", classCount)
        mlflow.log_param("learning_rate", learningRate)
        mlflow.log_param("iterations", iterations)

        if useCrossValidation:
            mlflow.log_param("k_folds", kFolds)
            mlflow.log_metric("mean_test_accuracy", results["mean_accuracy"])
            mlflow.log_metric("mean_macro_f1", results["mean_macro_f1"])
            mlflow.log_metric("mean_macro_precision", results["mean_macro_precision"])
            mlflow.log_metric("mean_macro_recall", results["mean_macro_recall"])

            for splitNumber, value in enumerate(results["fold_macro_precision"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_macro_precision", value)

            for splitNumber, value in enumerate(results["fold_macro_recall"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_macro_recall", value)

            for splitNumber, value in enumerate(results["fold_accuracies"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_accuracy", value)

            for splitNumber, value in enumerate(results["fold_macro_f1"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_macro_f1", value)

            for splitNumber in range(1, len(splits) + 1):
                mlflow.log_artifact(f"logistic_confusion_matrix_split_{splitNumber}.png")
                mlflow.log_artifact(f"logistic_metrics_split_{splitNumber}.png")
        else:
            mlflow.log_metric("test_accuracy", float(results["test_accuracy"]))
            mlflow.log_metric("macro_precision", float(results["metrics"]["macro_precision"]))
            mlflow.log_metric("macro_recall", float(results["metrics"]["macro_recall"]))
            mlflow.log_metric("macro_f1", float(results["metrics"]["macro_f1"]))

            for classIndex in range(classCount):
                mlflow.log_metric(f"class_{classIndex}_precision", float(results["metrics"]["precision"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_recall", float(results["metrics"]["recall"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_f1", float(results["metrics"]["f1"][classIndex]))

            mlflow.log_artifact("logistic_coefficients.txt")
            mlflow.log_artifact(results["confusion_plot_path"])
            mlflow.log_artifact(results["metrics_plot_path"])

    if useCrossValidation:
        print("Logistic regression mean CV accuracy:", results["mean_accuracy"])
        print("Logistic regression mean CV macro F1:", results["mean_macro_f1"])
        print("Logistic regression mean CV macro precision:", results["mean_macro_precision"])
        print("Logistic regression mean CV macro recall:", results["mean_macro_recall"])
    else:
        print("Logistic regression test accuracy:", float(results["test_accuracy"]))
        print("Logistic regression test macro F1:", float(results["metrics"]["macro_f1"]))
        print("Logistic regression test macro precision:", float(results["metrics"]["macro_precision"]))
        print("Logistic regression test macro recall:", float(results["metrics"]["macro_recall"]))

    return results
