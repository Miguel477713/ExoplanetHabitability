import numpy
import jax
import mlflow

from Models.LogisticRegression import LogisticRegressionCoefficients, EstimateWithLogisticRegression
from Utils.JaxUtils import AddIntercept


def RunLogisticRegressionPipeline(X, Y, splits, classCount, useCrossValidation, kFolds):
    learningRate = 0.01
    iterations = 2000

    results = {
        "fold_accuracies": [],
    }

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

        results["fold_accuracies"].append(float(logisticTestAccuracy))

        if not useCrossValidation and splitNumber == 1:
            results["coefficients"] = logisticRegressionB
            results["test_predictions"] = logisticTestPredictions
            results["test_accuracy"] = logisticTestAccuracy
            numpy.savetxt("logistic_coefficients.txt", numpy.asarray(logisticRegressionB), fmt="%.10f")

        print(f"Logistic regression split {splitNumber} accuracy:", float(logisticTestAccuracy))

    results["mean_accuracy"] = float(sum(results["fold_accuracies"]) / len(results["fold_accuracies"]))

    runName = "logistic_regression_cv" if useCrossValidation else "logistic_regression"

    with mlflow.start_run(run_name=runName):
        mlflow.log_param("model", "multiclass_logistic_regression")
        mlflow.log_param("class_count", classCount)
        mlflow.log_param("learning_rate", learningRate)
        mlflow.log_param("iterations", iterations)

        if useCrossValidation:
            mlflow.log_param("k_folds", kFolds)
            mlflow.log_metric("mean_test_accuracy", results["mean_accuracy"])

            for splitNumber, value in enumerate(results["fold_accuracies"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_accuracy", value)
        else:
            mlflow.log_metric("test_accuracy", float(results["test_accuracy"]))
            mlflow.log_artifact("logistic_coefficients.txt")

    if useCrossValidation:
        print("Logistic regression mean CV accuracy:", results["mean_accuracy"])
    else:
        print("Logistic regression test accuracy:", float(results["test_accuracy"]))

    return results
