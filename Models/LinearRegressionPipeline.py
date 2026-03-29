import numpy
import jax
import mlflow

from Models.LinearRegression import LinearRegressionCoefficients, EstimateWithLinearRegression
from Utils.JaxUtils import AddIntercept
from Metrics.Metrics import ConfusionMatrix, ClassificationMetrics, SaveConfusionMatrixPlot, SavePerClassMetricsPlot, PrintMetrics


def RunLinearRegressionPipeline(X, Y, splits, classCount, useCrossValidation, kFolds):
    results = {
    "fold_accuracies": [],
    "fold_mses": [],
    "fold_macro_f1": [],
    "fold_macro_precision": [],
    "fold_macro_recall": [],
    "mean_accuracy": None,
    "mean_mse": None,
    "mean_macro_f1": None,
    "mean_macro_precision": None,
    "mean_macro_recall": None,
    "test_accuracy": None,
    "test_mse": None,
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

        results["fold_accuracies"].append(float(linearTestAccuracy))
        results["fold_mses"].append(float(linearTestMSE))
        results["fold_macro_f1"].append(float(metrics["macro_f1"]))
        results["fold_macro_precision"].append(float(metrics["macro_precision"]))
        results["fold_macro_recall"].append(float(metrics["macro_recall"]))

        confusionPlotPath = f"linear_confusion_matrix_split_{splitNumber}.png"
        metricsPlotPath = f"linear_metrics_split_{splitNumber}.png"

        SaveConfusionMatrixPlot(
            confusionMatrix,
            classLabels,
            confusionPlotPath,
            f"Linear Regression Confusion Matrix - Split {splitNumber}",
        )
        SavePerClassMetricsPlot(
            metrics,
            classLabels,
            metricsPlotPath,
            f"Linear Regression Precision/Recall/F1 - Split {splitNumber}",
        )

        print(f"Linear regression split {splitNumber} accuracy:", float(linearTestAccuracy))
        print(f"Linear regression split {splitNumber} mse:", float(linearTestMSE))
        PrintMetrics(metrics)

        if not useCrossValidation and splitNumber == 1:
            results["coefficients"] = linearRegressionB
            results["test_predictions"] = linearTestPredictions
            results["test_accuracy"] = linearTestAccuracy
            results["test_mse"] = linearTestMSE
            results["confusion_matrix"] = confusionMatrix
            results["metrics"] = metrics
            results["confusion_plot_path"] = confusionPlotPath
            results["metrics_plot_path"] = metricsPlotPath
            numpy.savetxt("linear_coefficients.txt", numpy.asarray(linearRegressionB), fmt="%.10f")

    results["mean_accuracy"] = float(sum(results["fold_accuracies"]) / len(results["fold_accuracies"]))
    results["mean_mse"] = float(sum(results["fold_mses"]) / len(results["fold_mses"]))
    results["mean_macro_f1"] = float(sum(results["fold_macro_f1"]) / len(results["fold_macro_f1"]))
    results["mean_macro_precision"] = float(sum(results["fold_macro_precision"]) / len(results["fold_macro_precision"]))
    results["mean_macro_recall"] = float(sum(results["fold_macro_recall"]) / len(results["fold_macro_recall"]))

    runName = "linear_regression_cv" if useCrossValidation else "linear_regression"

    with mlflow.start_run(run_name=runName, nested=True):
        mlflow.log_param("model", "linear_regression_classifier")

        if useCrossValidation:
            mlflow.log_param("k_folds", kFolds)
            mlflow.log_metric("mean_test_accuracy", results["mean_accuracy"])
            mlflow.log_metric("mean_test_mse", results["mean_mse"])
            mlflow.log_metric("mean_macro_f1", results["mean_macro_f1"])
            mlflow.log_metric("mean_macro_precision", results["mean_macro_precision"])
            mlflow.log_metric("mean_macro_recall", results["mean_macro_recall"])

            for splitNumber, value in enumerate(results["fold_accuracies"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_accuracy", value)

            for splitNumber, value in enumerate(results["fold_mses"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_mse", value)

            for splitNumber, value in enumerate(results["fold_macro_f1"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_macro_f1", value)

            for splitNumber, value in enumerate(results["fold_macro_precision"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_macro_precision", value)

            for splitNumber, value in enumerate(results["fold_macro_recall"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_macro_recall", value)

            for splitNumber in range(1, len(splits) + 1):
                mlflow.log_artifact(f"linear_confusion_matrix_split_{splitNumber}.png")
                mlflow.log_artifact(f"linear_metrics_split_{splitNumber}.png")
        else:
            mlflow.log_param("feature_count_with_intercept", int(XTraining.shape[1]))
            mlflow.log_metric("test_accuracy", float(results["test_accuracy"]))
            mlflow.log_metric("test_mse", float(results["test_mse"]))
            mlflow.log_metric("macro_precision", float(results["metrics"]["macro_precision"]))
            mlflow.log_metric("macro_recall", float(results["metrics"]["macro_recall"]))
            mlflow.log_metric("macro_f1", float(results["metrics"]["macro_f1"]))

            for classIndex in range(classCount):
                mlflow.log_metric(f"class_{classIndex}_precision", float(results["metrics"]["precision"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_recall", float(results["metrics"]["recall"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_f1", float(results["metrics"]["f1"][classIndex]))

            mlflow.log_artifact("linear_coefficients.txt")
            mlflow.log_artifact(results["confusion_plot_path"])
            mlflow.log_artifact(results["metrics_plot_path"])

    if useCrossValidation:
        print("Linear regression mean CV accuracy:", results["mean_accuracy"])
        print("Linear regression mean CV mse:", results["mean_mse"])
        print("Linear regression mean CV macro F1:", results["mean_macro_f1"])
        print("Linear regression mean CV macro precision:", results["mean_macro_precision"])
        print("Linear regression mean CV macro recall:", results["mean_macro_recall"])
    else:
        print("Linear regression test accuracy:", float(results["test_accuracy"]))
        print("Linear regression test mse:", float(results["test_mse"]))
        print("Linear regression test macro F1:", float(results["metrics"]["macro_f1"]))
        print("Linear regression test macro precision:", float(results["metrics"]["macro_precision"]))
        print("Linear regression test macro recall:", float(results["metrics"]["macro_recall"]))

    return results
