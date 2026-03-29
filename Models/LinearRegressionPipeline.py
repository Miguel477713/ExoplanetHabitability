import numpy
import jax
import mlflow

from Models.LinearRegression import LinearRegressionCoefficients, EstimateWithLinearRegression
from Utils.JaxUtils import AddIntercept


def RunLinearRegressionPipeline(X, Y, splits, useCrossValidation, kFolds):
    results = {
        "fold_accuracies": [],
        "fold_mses": [],
    }

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

        results["fold_accuracies"].append(float(linearTestAccuracy))
        results["fold_mses"].append(float(linearTestMSE))

        if not useCrossValidation and splitNumber == 1:
            results["coefficients"] = linearRegressionB
            results["test_predictions"] = linearTestPredictions
            results["test_accuracy"] = linearTestAccuracy
            results["test_mse"] = linearTestMSE
            numpy.savetxt("linear_coefficients.txt", numpy.asarray(linearRegressionB), fmt="%.10f")

        print(f"Linear regression split {splitNumber} accuracy:", float(linearTestAccuracy))
        print(f"Linear regression split {splitNumber} mse:", float(linearTestMSE))

    results["mean_accuracy"] = float(sum(results["fold_accuracies"]) / len(results["fold_accuracies"]))
    results["mean_mse"] = float(sum(results["fold_mses"]) / len(results["fold_mses"]))

    runName = "linear_regression_cv" if useCrossValidation else "linear_regression"

    with mlflow.start_run(run_name=runName, nested=True):
        mlflow.log_param("model", "linear_regression_classifier")

        if useCrossValidation:
            mlflow.log_param("k_folds", kFolds)
            mlflow.log_metric("mean_test_accuracy", results["mean_accuracy"])
            mlflow.log_metric("mean_test_mse", results["mean_mse"])

            for splitNumber, value in enumerate(results["fold_accuracies"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_accuracy", value)

            for splitNumber, value in enumerate(results["fold_mses"], start=1):
                mlflow.log_metric(f"split_{splitNumber}_mse", value)
        else:
            mlflow.log_param("feature_count_with_intercept", int(XTraining.shape[1]))
            mlflow.log_metric("test_accuracy", float(results["test_accuracy"]))
            mlflow.log_metric("test_mse", float(results["test_mse"]))
            mlflow.log_artifact("linear_coefficients.txt")

    if useCrossValidation:
        print("Linear regression mean CV accuracy:", results["mean_accuracy"])
        print("Linear regression mean CV mse:", results["mean_mse"])
    else:
        print("Linear regression test accuracy:", float(results["test_accuracy"]))
        print("Linear regression test mse:", float(results["test_mse"]))

    return results
