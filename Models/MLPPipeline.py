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



def RunMLPPipeline(X, Y, splits, classCount):
    hiddenUnitCount = 12
    learningRate = 0.05
    maxEpochs = 1000
    tolerance = 1e-6

    trainIndices, testIndices = splits[0]

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
        randomSeed=0,
    )
    mlp.Fit(XTraining, YTraining)

    mlpPredictions = mlp.Predict(XTest)
    mlpAccuracy = numpy.mean(mlpPredictions == YTest)

    confusionMatrix = ConfusionMatrix(YTest, mlpPredictions, classCount)
    metrics = ClassificationMetrics(confusionMatrix)
    classLabels = [str(i) for i in range(classCount)]

    confusionPlotPath = "mlp_confusion_matrix.png"
    metricsPlotPath = "mlp_metrics.png"
    errorPlotPath = "mlp_training_error.png"

    SaveConfusionMatrixPlot(
        confusionMatrix,
        classLabels,
        confusionPlotPath,
        "MLP Confusion Matrix",
    )
    SavePerClassMetricsPlot(
        metrics,
        classLabels,
        metricsPlotPath,
        "MLP Precision/Recall/F1",
    )
    SaveTrainingErrorPlot(
        mlp.trainingErrors,
        errorPlotPath,
        "MLP Training Error",
    )

    print("MLP test accuracy:", float(mlpAccuracy))
    print("MLP epochs:", int(mlp.epochCount))
    PrintMetrics(metrics)

    results = {
        "model": mlp,
        "test_predictions": mlpPredictions,
        "test_accuracy": float(mlpAccuracy),
        "confusion_matrix": confusionMatrix,
        "metrics": metrics,
        "confusion_plot_path": confusionPlotPath,
        "metrics_plot_path": metricsPlotPath,
        "training_error_plot_path": errorPlotPath,
        "epoch_count": int(mlp.epochCount),
        "feature_mean": mean,
        "feature_std": std,
    }

    with mlflow.start_run(run_name="mlp", nested=True):
        mlflow.log_param("model", "mlp_one_hidden_layer_classifier")
        mlflow.log_param("class_count", classCount)
        mlflow.log_param("hidden_units", hiddenUnitCount)
        mlflow.log_param("learning_rate", learningRate)
        mlflow.log_param("max_epochs", maxEpochs)
        mlflow.log_param("tolerance", tolerance)
        mlflow.log_param("input_standardization", True)
        mlflow.log_metric("test_accuracy", results["test_accuracy"])
        mlflow.log_metric("macro_precision", float(metrics["macro_precision"]))
        mlflow.log_metric("macro_recall", float(metrics["macro_recall"]))
        mlflow.log_metric("macro_f1", float(metrics["macro_f1"]))
        mlflow.log_metric("epoch_count", float(results["epoch_count"]))

        for classIndex in range(classCount):
            mlflow.log_metric(f"class_{classIndex}_precision", float(metrics["precision"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_recall", float(metrics["recall"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_f1", float(metrics["f1"][classIndex]))

        mlflow.log_artifact(confusionPlotPath)
        mlflow.log_artifact(metricsPlotPath)
        mlflow.log_artifact(errorPlotPath)

    return results
