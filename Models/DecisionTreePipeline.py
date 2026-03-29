import numpy
import mlflow

from Models.DecisionTree import DecisionTreeClassifierScratch, TreeToRules
from Metrics.Metrics import ConfusionMatrix, ClassificationMetrics, SaveConfusionMatrixPlot, SavePerClassMetricsPlot, PrintMetrics


def RunDecisionTreePipeline(X, Y, splits, classCount, useCrossValidation, kFolds, featureNames):
    maxDepth = 8
    minSamplesSplit = 10
    minGain = 1e-6

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

        results["fold_accuracies"].append(float(decisionTreeAccuracy))
        results["fold_macro_f1"].append(float(metrics["macro_f1"]))
        results["fold_macro_precision"].append(float(metrics["macro_precision"]))
        results["fold_macro_recall"].append(float(metrics["macro_recall"]))

        confusionPlotPath = f"decision_tree_confusion_matrix_split_{splitNumber}.png"
        metricsPlotPath = f"decision_tree_metrics_split_{splitNumber}.png"
        rulesPath = f"decision_tree_rules_split_{splitNumber}.txt"

        SaveConfusionMatrixPlot(
            confusionMatrix,
            classLabels,
            confusionPlotPath,
            f"Decision Tree Confusion Matrix - Split {splitNumber}",
        )
        SavePerClassMetricsPlot(
            metrics,
            classLabels,
            metricsPlotPath,
            f"Decision Tree Precision/Recall/F1 - Split {splitNumber}",
        )

        with open(rulesPath, "w", encoding="utf-8") as file:
            file.write(TreeToRules(tree.root, featureNames))

        print(f"Decision tree split {splitNumber} accuracy:", float(decisionTreeAccuracy))
        print(f"Decision tree split {splitNumber} depth:", tree.Depth())
        print(f"Decision tree split {splitNumber} leaves:", tree.LeafCount())
        PrintMetrics(metrics)

        if not useCrossValidation and splitNumber == 1:
            results["model"] = tree
            results["test_predictions"] = decisionTreePredictions
            results["test_accuracy"] = float(decisionTreeAccuracy)
            results["confusion_matrix"] = confusionMatrix
            results["metrics"] = metrics
            results["confusion_plot_path"] = confusionPlotPath
            results["metrics_plot_path"] = metricsPlotPath
            results["rules_path"] = rulesPath
            results["tree_depth"] = tree.Depth()
            results["leaf_count"] = tree.LeafCount()

    results["mean_accuracy"] = float(sum(results["fold_accuracies"]) / len(results["fold_accuracies"]))
    results["mean_macro_f1"] = float(sum(results["fold_macro_f1"]) / len(results["fold_macro_f1"]))
    results["mean_macro_precision"] = float(sum(results["fold_macro_precision"]) / len(results["fold_macro_precision"]))
    results["mean_macro_recall"] = float(sum(results["fold_macro_recall"]) / len(results["fold_macro_recall"]))

    runName = "decision_tree_cv" if useCrossValidation else "decision_tree"

    with mlflow.start_run(run_name=runName, nested=True):
        mlflow.log_param("model", "decision_tree_classifier")
        mlflow.log_param("class_count", classCount)
        mlflow.log_param("max_depth", maxDepth)
        mlflow.log_param("min_samples_split", minSamplesSplit)
        mlflow.log_param("min_gain", minGain)

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
                mlflow.log_artifact(f"decision_tree_confusion_matrix_split_{splitNumber}.png")
                mlflow.log_artifact(f"decision_tree_metrics_split_{splitNumber}.png")
                mlflow.log_artifact(f"decision_tree_rules_split_{splitNumber}.txt")
        else:
            mlflow.log_metric("test_accuracy", float(results["test_accuracy"]))
            mlflow.log_metric("macro_precision", float(results["metrics"]["macro_precision"]))
            mlflow.log_metric("macro_recall", float(results["metrics"]["macro_recall"]))
            mlflow.log_metric("macro_f1", float(results["metrics"]["macro_f1"]))
            mlflow.log_metric("tree_depth", float(results["tree_depth"]))
            mlflow.log_metric("leaf_count", float(results["leaf_count"]))

            for classIndex in range(classCount):
                mlflow.log_metric(f"class_{classIndex}_precision", float(results["metrics"]["precision"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_recall", float(results["metrics"]["recall"][classIndex]))
                mlflow.log_metric(f"class_{classIndex}_f1", float(results["metrics"]["f1"][classIndex]))

            mlflow.log_artifact(results["confusion_plot_path"])
            mlflow.log_artifact(results["metrics_plot_path"])
            mlflow.log_artifact(results["rules_path"])

    if useCrossValidation:
        print("Decision tree mean CV accuracy:", results["mean_accuracy"])
        print("Decision tree mean CV macro F1:", results["mean_macro_f1"])
        print("Decision tree mean CV macro precision:", results["mean_macro_precision"])
        print("Decision tree mean CV macro recall:", results["mean_macro_recall"])
    else:
        print("Decision tree test accuracy:", float(results["test_accuracy"]))
        print("Decision tree test macro F1:", float(results["metrics"]["macro_f1"]))
        print("Decision tree test macro precision:", float(results["metrics"]["macro_precision"]))
        print("Decision tree test macro recall:", float(results["metrics"]["macro_recall"]))

    return results
