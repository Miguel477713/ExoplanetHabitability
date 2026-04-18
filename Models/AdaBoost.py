import math
import numpy
import jax

from Utils.JaxUtils import AddIntercept
from Models.LinearRegression import LinearRegressionCoefficients, EstimateWithLinearRegression
from Models.LogisticRegression import LogisticRegressionCoefficients, EstimateWithLogisticRegression
from Models.DecisionTree import DecisionTreeClassifierScratch


class LinearRegressionClassifierWrapper:
    def __init__(self):
        self.B = None

    def Fit(self, X, y):
        Xj = jax.numpy.array(numpy.asarray(X, dtype=numpy.float64))
        yj = jax.numpy.array(numpy.asarray(y).reshape(-1, 1), dtype=jax.numpy.float64)
        Xj = AddIntercept(Xj)
        self.B = LinearRegressionCoefficients(Xj, yj)
        return self

    def Predict(self, X):
        Xj = AddIntercept(jax.numpy.array(numpy.asarray(X, dtype=numpy.float64)))
        predictions = []
        for rowIndex in range(Xj.shape[0]):
            prediction = EstimateWithLinearRegression(
                self.B,
                Xj[rowIndex].reshape(-1, 1),
                isClassification=True,
            )
            predictions.append(int(prediction))
        return numpy.asarray(predictions, dtype=int).reshape(-1, 1)


class LogisticRegressionClassifierWrapper:
    def __init__(self, classCount, learningRate=0.01, iterations=400):
        self.classCount = int(classCount)
        self.learningRate = float(learningRate)
        self.iterations = int(iterations)
        self.B = None

    def Fit(self, X, y):
        Xj = jax.numpy.array(numpy.asarray(X, dtype=numpy.float64))
        yj = jax.numpy.array(numpy.asarray(y).reshape(-1, 1), dtype=int)
        Xj = AddIntercept(Xj)
        self.B = LogisticRegressionCoefficients(
            Xj,
            yj,
            classCount=self.classCount,
            learningRate=self.learningRate,
            iterations=self.iterations,
        )
        return self

    def Predict(self, X):
        Xj = AddIntercept(jax.numpy.array(numpy.asarray(X, dtype=numpy.float64)))
        predictions = []
        for rowIndex in range(Xj.shape[0]):
            prediction = EstimateWithLogisticRegression(
                self.B,
                Xj[rowIndex].reshape(-1, 1),
            )
            predictions.append(int(prediction))
        return numpy.asarray(predictions, dtype=int).reshape(-1, 1)


class DecisionTreeClassifierWrapper:
    def __init__(self, classCount, maxDepth=1, minSamplesSplit=2, minGain=1e-6):
        self.model = DecisionTreeClassifierScratch(
            classCount=classCount,
            maxDepth=maxDepth,
            minSamplesSplit=minSamplesSplit,
            minGain=minGain,
        )

    def Fit(self, X, y):
        self.model.Fit(
            numpy.asarray(X, dtype=numpy.float64),
            numpy.asarray(y).reshape(-1).astype(int),
        )
        return self

    def Predict(self, X):
        return self.model.Predict(numpy.asarray(X, dtype=numpy.float64))


def CreateBaseEstimator(baseModelName, classCount):
    baseModelName = str(baseModelName).lower()

    if baseModelName == "linear":
        return LinearRegressionClassifierWrapper()

    if baseModelName == "logistic":
        return LogisticRegressionClassifierWrapper(
            classCount=classCount,
            learningRate=0.01,
            iterations=350,
        )

    if baseModelName == "decision_tree":
        return DecisionTreeClassifierWrapper(
            classCount=classCount,
            maxDepth=1,
            minSamplesSplit=2,
            minGain=1e-6,
        )

    raise ValueError(f"Unsupported AdaBoost base model: {baseModelName}")


class AdaBoostSAMMEClassifier:
    def __init__(self, classCount, baseModelName, nEstimators=20, learningRate=1.0, randomState=0):
        self.classCount = int(classCount)
        self.baseModelName = str(baseModelName).lower()
        self.nEstimators = int(nEstimators)
        self.learningRate = float(learningRate)
        self.randomState = int(randomState)
        self.estimators = []
        self.estimatorWeights = []
        self.estimatorErrors = []
        self.weightHistory = []
        self.majorityClass_ = 0

    def Fit(self, X, y):
        X = numpy.asarray(X, dtype=numpy.float64)
        y = numpy.asarray(y).reshape(-1).astype(int)
        sampleCount = X.shape[0]
        rng = numpy.random.default_rng(self.randomState)

        counts = numpy.bincount(y, minlength=self.classCount)
        self.majorityClass_ = int(numpy.argmax(counts))

        weights = numpy.full(sampleCount, 1.0 / sampleCount, dtype=numpy.float64)
        self.weightHistory = [weights.copy()]

        for estimatorIndex in range(self.nEstimators):
            sampledIndices = rng.choice(
                sampleCount,
                size=sampleCount,
                replace=True,
                p=weights,
            )

            estimator = CreateBaseEstimator(self.baseModelName, self.classCount)
            estimator.Fit(X[sampledIndices], y[sampledIndices])

            predictions = estimator.Predict(X).reshape(-1).astype(int)
            incorrect = predictions != y

            error = float(numpy.sum(weights * incorrect))
            error = min(max(error, 1e-12), 1.0 - 1e-12)

            if error >= 1.0 - (1.0 / self.classCount):
                continue

            alpha = self.learningRate * (
                math.log((1.0 - error) / error) + math.log(max(1, self.classCount - 1))
            )

            self.estimators.append(estimator)
            self.estimatorWeights.append(float(alpha))
            self.estimatorErrors.append(float(error))

            if error <= 1e-12:
                break

            weights = weights * numpy.exp(alpha * incorrect.astype(numpy.float64))
            weights = weights / numpy.sum(weights)
            self.weightHistory.append(weights.copy())

        return self

    def Predict(self, X):
        X = numpy.asarray(X, dtype=numpy.float64)
        sampleCount = X.shape[0]

        if len(self.estimators) == 0:
            return numpy.full((sampleCount, 1), self.majorityClass_, dtype=int)

        votes = numpy.zeros((sampleCount, self.classCount), dtype=numpy.float64)

        for estimator, alpha in zip(self.estimators, self.estimatorWeights):
            predictions = estimator.Predict(X).reshape(-1).astype(int)
            for classIndex in range(self.classCount):
                votes[:, classIndex] += alpha * (predictions == classIndex)

        return numpy.argmax(votes, axis=1).astype(int).reshape(-1, 1)


def BuildAdaBoostModel(baseModelName, XTraining, classCount, randomState=0):
    baseModelName = str(baseModelName).lower()

    if baseModelName == "linear":
        return AdaBoostSAMMEClassifier(
            classCount=classCount,
            baseModelName="linear",
            nEstimators=15,
            learningRate=0.6,
            randomState=randomState,
        )

    if baseModelName == "logistic":
        return AdaBoostSAMMEClassifier(
            classCount=classCount,
            baseModelName="logistic",
            nEstimators=12,
            learningRate=0.7,
            randomState=randomState,
        )

    if baseModelName == "decision_tree":
        return AdaBoostSAMMEClassifier(
            classCount=classCount,
            baseModelName="decision_tree",
            nEstimators=25,
            learningRate=1.0,
            randomState=randomState,
        )

    raise ValueError(f"Unsupported AdaBoost base model: {baseModelName}")