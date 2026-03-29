import jax


def Softmax(Z):
    # to stabilize large exponents, does not affect probabilities since it affects all classes equally
    shiftedZ = Z - jax.numpy.max(Z, axis=1, keepdims=True)
    expZ = jax.numpy.exp(shiftedZ)
    return expZ / jax.numpy.sum(expZ, axis=1, keepdims=True)


def OneHotEncode(y, classCount):
    yFlat = jax.numpy.ravel(y).astype(int)
    return jax.numpy.eye(classCount, dtype=jax.numpy.float64)[yFlat]


def EstimateWithLogisticRegression(B, x):
    logits = x.T @ B
    probabilities = Softmax(logits.reshape(1, -1))
    return int(jax.numpy.argmax(probabilities, axis=1)[0])


def LogisticRegressionCoefficients(X, y, classCount, learningRate=0.01, iterations=2000):
    B = jax.numpy.zeros((X.shape[1], classCount), dtype=jax.numpy.float64)
    sampleCount = X.shape[0]
    yOneHot = OneHotEncode(y, classCount)

    for _ in range(iterations):
        scores = X @ B
        probabilities = Softmax(scores)
        gradient = (X.T @ (probabilities - yOneHot)) / sampleCount
        B = B - learningRate * gradient

    return B
