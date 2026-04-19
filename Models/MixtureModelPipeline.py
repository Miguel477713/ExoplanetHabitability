import numpy
import mlflow

from Models.MixtureModel import MixtureModelClassifier
from Metrics.Metrics import ConfusionMatrix, ClassificationMetrics, SaveConfusionMatrixPlot, SavePerClassMetricsPlot, PrintMetrics


def SaveMixtureModelSummary(model, outputPath, title):
    with open(outputPath, "w", encoding="utf-8") as file:
        file.write(title + "\n")
        file.write("Experts\n")
        for expertName, accuracy in zip(model.ExpertNames, model.ExpertTrainingAccuracy):
            file.write(f"{expertName}: training_accuracy={float(accuracy):.10f}\n")
        file.write("\n")
        file.write("Gating model\n")
        file.write(f"learning_rate={model.GatingLearningRate}\n")
        file.write(f"iterations={model.GatingIterations}\n")
        file.write(f"hidden_unit_count={model.HiddenUnitCount}\n")
        file.write(f"mlp_learning_rate={model.MlpLearningRate}\n")
        file.write(f"mlp_max_epochs={model.MlpMaxEpochs}\n")
        file.write(f"mlp_tolerance={model.MlpTolerance}\n")


def RunMixtureModelPipeline(X, Y, splits, classCount):
    trainIndices, testIndices = splits[0]

    XTraining = numpy.asarray(X[trainIndices, :], dtype=numpy.float64)
    YTraining = numpy.asarray(Y[trainIndices]).reshape(-1).astype(int)
    XTest = numpy.asarray(X[testIndices, :], dtype=numpy.float64)
    YTest = numpy.asarray(Y[testIndices]).reshape(-1, 1).astype(int)

    model = MixtureModelClassifier(classCount=classCount)
    model.Fit(XTraining, YTraining)

    predictionDetails = model.PredictWithDetails(XTest)
    mixturePredictions = predictionDetails["Predictions"]
    mixtureAccuracy = numpy.mean(mixturePredictions == YTest)

    confusionMatrix = ConfusionMatrix(YTest, mixturePredictions, classCount)
    metrics = ClassificationMetrics(confusionMatrix)
    classLabels = [str(i) for i in range(classCount)]

    confusionPlotPath = "mixture_model_confusion_matrix.png"
    metricsPlotPath = "mixture_model_metrics.png"
    summaryPath = "mixture_model_summary.txt"

    SaveConfusionMatrixPlot(
        confusionMatrix,
        classLabels,
        confusionPlotPath,
        "Mixture Model Confusion Matrix",
    )
    SavePerClassMetricsPlot(
        metrics,
        classLabels,
        metricsPlotPath,
        "Mixture Model Precision/Recall/F1",
    )
    SaveMixtureModelSummary(
        model,
        summaryPath,
        "Mixture Model summary",
    )

    print("Mixture model test accuracy:", float(mixtureAccuracy))
    PrintMetrics(metrics)

    results = {
        "model": model,
        "test_predictions": mixturePredictions,
        "test_accuracy": float(mixtureAccuracy),
        "confusion_matrix": confusionMatrix,
        "metrics": metrics,
        "confusion_plot_path": confusionPlotPath,
        "metrics_plot_path": metricsPlotPath,
        "summary_path": summaryPath,
        "expert_names": list(model.ExpertNames),
        "expert_training_accuracy": [float(value) for value in model.ExpertTrainingAccuracy],
    }

    with mlflow.start_run(run_name="mixture_model", nested=True):
        mlflow.log_param("model", "mixture_of_experts_classifier")
        mlflow.log_param("class_count", classCount)
        mlflow.log_param("expert_count", len(model.ExpertNames))
        mlflow.log_param("gating_learning_rate", model.GatingLearningRate)
        mlflow.log_param("gating_iterations", model.GatingIterations)
        mlflow.log_param("mlp_hidden_units", model.HiddenUnitCount)
        mlflow.log_metric("test_accuracy", results["test_accuracy"])
        mlflow.log_metric("macro_precision", float(metrics["macro_precision"]))
        mlflow.log_metric("macro_recall", float(metrics["macro_recall"]))
        mlflow.log_metric("macro_f1", float(metrics["macro_f1"]))

        for expertName, expertAccuracy in zip(model.ExpertNames, results["expert_training_accuracy"]):
            mlflow.log_metric(f"{expertName}_training_accuracy", float(expertAccuracy))

        for classIndex in range(classCount):
            mlflow.log_metric(f"class_{classIndex}_precision", float(metrics["precision"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_recall", float(metrics["recall"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_f1", float(metrics["f1"][classIndex]))

        mlflow.log_artifact(confusionPlotPath)
        mlflow.log_artifact(metricsPlotPath)
        mlflow.log_artifact(summaryPath)

    return results
