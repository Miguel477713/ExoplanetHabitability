import jax, numpy


def EstimateWithLinearRegression(B, x, isClassification=False):
    temp = x.T @ B

    if isClassification:
        predictedValue = float(temp.squeeze())

        if numpy.isnan(predictedValue) or numpy.isinf(predictedValue):
            raise ValueError("Linear regression prediction became NaN or Inf")

        if predictedValue < 0.5:
            return 0
        if predictedValue < 1.5:
            return 1
        return 2

    return temp


def LinearRegressionCoefficients(X, y, ridgeLambda=1e-6):
    identity = jax.numpy.eye(X.shape[1], dtype=jax.numpy.float64)
    B = jax.numpy.linalg.solve(X.T @ X + ridgeLambda * identity, X.T @ y)
    #B = jax.numpy.linalg.solve(X.T @ X, X.T @ y)
    return B
