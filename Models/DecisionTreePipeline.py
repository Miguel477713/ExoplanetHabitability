import numpy
import mlflow

from Models.DecisionTree import DecisionTreeClassifierScratch, TreeToRules
from Metrics.Metrics import ConfusionMatrix, ClassificationMetrics, SaveConfusionMatrixPlot, SavePerClassMetricsPlot, PrintMetrics


def RunDecisionTreePipeline(X, Y, splits, classCount, featureNames):
    maxDepth = 8
    minSamplesSplit = 10
    minGain = 1e-6

    trainIndices, testIndices = splits[0]

    XTraining = numpy.asarray(X[trainIndices, :], dtype=numpy.float64)
    YTraining = numpy.asarray(Y[trainIndices]).reshape(-1).astype(int)
    XTest = numpy.asarray(X[testIndices, :], dtype=numpy.float64)
    YTest = numpy.asarray(Y[testIndices]).reshape(-1, 1).astype(int)

    tree = DecisionTreeClassifierScratch(
        classCount=classCount,
        maxDepth=maxDepth,
        minSamplesSplit=minSamplesSplit,
        minGain=minGain,
    )
    tree.Fit(XTraining, YTraining)

    decisionTreePredictions = tree.Predict(XTest)
    decisionTreeAccuracy = numpy.mean(decisionTreePredictions == YTest)

    confusionMatrix = ConfusionMatrix(YTest, decisionTreePredictions, classCount)
    metrics = ClassificationMetrics(confusionMatrix)
    classLabels = [str(i) for i in range(classCount)]

    confusionPlotPath = "decision_tree_confusion_matrix.png"
    metricsPlotPath = "decision_tree_metrics.png"
    rulesPath = "decision_tree_rules.txt"

    SaveConfusionMatrixPlot(
        confusionMatrix,
        classLabels,
        confusionPlotPath,
        "Decision Tree Confusion Matrix",
    )
    SavePerClassMetricsPlot(
        metrics,
        classLabels,
        metricsPlotPath,
        "Decision Tree Precision/Recall/F1",
    )

    with open(rulesPath, "w", encoding="utf-8") as file:
        file.write(TreeToRules(tree.root, featureNames))

    print("Decision tree test accuracy:", float(decisionTreeAccuracy))
    print("Decision tree test depth:", tree.Depth())
    print("Decision tree test leaves:", tree.LeafCount())
    PrintMetrics(metrics)

    results = {
        "model": tree,
        "test_predictions": decisionTreePredictions,
        "test_accuracy": float(decisionTreeAccuracy),
        "confusion_matrix": confusionMatrix,
        "metrics": metrics,
        "confusion_plot_path": confusionPlotPath,
        "metrics_plot_path": metricsPlotPath,
        "rules_path": rulesPath,
        "tree_depth": tree.Depth(),
        "leaf_count": tree.LeafCount(),
    }

    with mlflow.start_run(run_name="decision_tree", nested=True):
        mlflow.log_param("model", "decision_tree_classifier")
        mlflow.log_param("class_count", classCount)
        mlflow.log_param("max_depth", maxDepth)
        mlflow.log_param("min_samples_split", minSamplesSplit)
        mlflow.log_param("min_gain", minGain)
        mlflow.log_metric("test_accuracy", results["test_accuracy"])
        mlflow.log_metric("macro_precision", float(metrics["macro_precision"]))
        mlflow.log_metric("macro_recall", float(metrics["macro_recall"]))
        mlflow.log_metric("macro_f1", float(metrics["macro_f1"]))
        mlflow.log_metric("tree_depth", float(results["tree_depth"]))
        mlflow.log_metric("leaf_count", float(results["leaf_count"]))

        for classIndex in range(classCount):
            mlflow.log_metric(f"class_{classIndex}_precision", float(metrics["precision"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_recall", float(metrics["recall"][classIndex]))
            mlflow.log_metric(f"class_{classIndex}_f1", float(metrics["f1"][classIndex]))

        mlflow.log_artifact(confusionPlotPath)
        mlflow.log_artifact(metricsPlotPath)
        mlflow.log_artifact(rulesPath)

    return results
