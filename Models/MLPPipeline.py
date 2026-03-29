import numpy
import mlflow
import matplotlib.pyplot as plt

from Models.MLP import MLPClassifierScratch
from Metrics.Metrics import ConfusionMatrix, ClassificationMetrics, SaveConfusionMatrixPlot, SavePerClassMetricsPlot, PrintMetrics


def StandardizeWithTrainingStatistics(XTraining, XTest):
    XTraining = numpy.asarray(XTraining, dtype=numpy.float64)
    XTest = numpy.asarray(XTest, dtype=numpy.float64)

    mean = XTraining.mean(axis=0, keepdims=True)
    std = XTraining.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0

    XTrainingStandardized = (XTraining - mean) / std
    XTestStandardized = (XTest - mean) / std
    return XTrainingStandardized, XTestStandardized, mean, std


def SaveTrainingErrorPlot(trainingErrors, outputPath, title):
    epochs = numpy.arange(1, len(trainingErrors) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, numpy.asarray(trainingErrors, dtype=numpy.float64))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average quadratic error")
    fig.tight_layout()
    fig.savefig(outputPath, bbox_inches="tight")
    plt.close(fig)



def RunMLPPipeline(X, Y, splits, classCount, useCrossValidation, kFolds):
    hiddenUnitCount = 12
    learningRate = 0.05
    maxEpochs = 1000
    tolerance = 1e-6

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
        XTraining = numpy.asarray(X[trainIndices, :], dtype=numpy.float64)
        YTraining = numpy.asarray(Y[trainIndices]).reshape(-1).astype(int)
        XTest = numpy.asarray(X[testIndices, :], dtype=numpy.float64)
        YTest = numpy.asarray(Y[testIndices]).reshape(-1, 1).astype(int)

        XTraining, XTest, mean, std = StandardizeWithTrainingStatistics(XTraining, XTest)

        mlp = MLPClassifierScratch(
            inputSize=XTraining.shape[1],
            hiddenUnitCount=hiddenUnitCount,
            outputSize=classCount,
            learningRate=learningRate,
            maxEpochs=maxEpochs,
            tolerance=tolerance,
            randomSeed=splitNumber - 1,
        )
        mlp.Fit(XTraining, YTraining)

        mlpPredictions = mlp.Predict(XTest)
        mlpAccuracy = numpy.mean(mlpPredictions == YTest)

        confusionMatrix = ConfusionMatrix(YTest, mlpPredictions, classCount)
        metrics = ClassificationMetrics(confusionMatrix)

        results["fold_accuracies"].append(float(mlpAccuracy))
        results["fold_macro_f1"].append(float(metrics["macro_f1"]))
        results["fold_macro_precision"].append(float(metrics["macro_precision"]))
        results["fold_macro_recall"].append(float(metrics["macro_recall"]))

        confusionPlotPath = f"mlp_confusion_matrix_split_{splitNumber}.png"
        metricsPlotPath = f"mlp_metrics_split_{splitNumber}.png"
        errorPlotPath = f"mlp_training_error_split_{splitNumber}.png"

        SaveConfusionMatrixPlot(
            confusionMatrix,
            classLabels,
            confusionPlotPath,
            f"MLP Confusion Matrix - Split {splitNumber}",
        )
        SavePerClassMetricsPlot(
            metrics,
            classLabels,
            metricsPlotPath,
            f"MLP Precision/Recall/F1 - Split {splitNumber}",
        )
        SaveTrainingErrorPlot(
            mlp.trainingErrors,
            errorPlotPath,
            f"MLP Training Error - Split {splitNumber}",
        )

        print(f"MLP split {splitNumber} accuracy:", float(mlpAccuracy))
        print(f"MLP split {splitNumber} epochs:", int(mlp.epochCount))
        PrintMetrics(metrics)

        if not useCrossValidation and splitNumber == 1:
            results["model"] = mlp
            results["test_predictions"] = mlpPredictions
            results["test_accuracy"] = float(mlpAccuracy)
            results["confusion_matrix"] = confusionMatrix
            results["metrics"] = metrics
            results["confusion_plot_path"] = confusionPlotPath
            results["metrics_plot_path"] = metricsPlotPath
            results["training_error_plot_path"] = errorPlotPath
            results["epoch_count"] = int(mlp.epochCount)
            results["feature_mean"] = mean
            results["feature_std"] = std

    results["mean_accuracy"] = float(sum(results["fold_accuracies"]) / len(results["fold_accuracies"]))
    results["mean_macro_f1"] = float(sum(results["fold_macro_f1"]) / len(results["fold_macro_f1"]))
    results["mean_macro_precision"] = float(sum(results["fold_macro_precision"]) / len(results["fold_macro_precision"]))
    results["mean_macro_recall"] = float(sum(results["fold_macro_recall"]) / len(results["fold_macro_recall"]))

    runName = "mlp_cv" if useCrossValidation else "mlp"

    with mlflow.start_run(run_name=runName, nested=True):
        mlflow.log_param("model", "mlp_one_hidden_layer_classifier")
        mlflow.log_param("class_count", classCount)
        mlflow.log_param("hidden_units", hiddenUnitCount)
        mlflow.log_param("learning_rate", learningRate)
        mlflow.log_param("max_epochs", maxEpochs)
        mlflow.log_param("tolerance", tolerance)
        mlflow.log_param("input_standardization", True)

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
                mlflow.log_artifact(f"mlp_confusion_matrix_split_{splitNumber}.png")
                mlflow.log_artifact(f"mlp_metrics_split_{splitNumber}.png")
                mlflow.log_artifact(f"mlp_training_error_split_{splitNumber}.png")
        else:
            mlflow.log_metric("test_accuracy", float(results["test_accuracy"]))
            mlflow.log_metric("macro_precision", float(results["metrics"]["macro_precision"]))
            mlflow.log_metric("macro_recall", float(results["metrics"]["macro_recall"]))
            mlflow.log_metric("macro_f1", float(results["metrics"]["macro_f1"]))
            mlflow.log_metric("epoch_count", float(results["epoch_count"]))

            for classIndex in range(classCount):
                mlflow.log_metric(f"class_{classIndex}_precision", float(results["metrics"]["precision"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_recall", float(results["metrics"]["recall"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_f1", float(results["metrics"]["f1"][classIndex]))

            mlflow.log_artifact(results["confusion_plot_path"])
            mlflow.log_artifact(results["metrics_plot_path"])
            mlflow.log_artifact(results["training_error_plot_path"])

    if useCrossValidation:
        print("MLP mean CV accuracy:", results["mean_accuracy"])
        print("MLP mean CV macro F1:", results["mean_macro_f1"])
        print("MLP mean CV macro precision:", results["mean_macro_precision"])
        print("MLP mean CV macro recall:", results["mean_macro_recall"])
    else:
        print("MLP test accuracy:", float(results["test_accuracy"]))
        print("MLP test macro F1:", float(results["metrics"]["macro_f1"]))
        print("MLP test macro precision:", float(results["metrics"]["macro_precision"]))
        print("MLP test macro recall:", float(results["metrics"]["macro_recall"]))

    return results
