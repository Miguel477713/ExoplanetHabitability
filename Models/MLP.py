import numpy as np


def Sigmoid(x):
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class MLPClassifierScratch:
    def __init__(
        self,
        inputSize,
        hiddenUnitCount,
        outputSize,
        learningRate=0.05,
        maxEpochs=1000,
        tolerance=1e-6,
        randomSeed=0,
    ):
        self.inputSize = int(inputSize)
        self.hiddenUnitCount = int(hiddenUnitCount)
        self.outputSize = int(outputSize)
        self.learningRate = float(learningRate)
        self.maxEpochs = int(maxEpochs)
        self.tolerance = float(tolerance)
        self.randomSeed = int(randomSeed)

        generator = np.random.default_rng(self.randomSeed)
        self.WInputHidden = generator.normal(
            loc=0.0,
            scale=0.1,
            size=(self.inputSize, self.hiddenUnitCount),
        )
        self.bHidden = np.zeros((1, self.hiddenUnitCount), dtype=np.float64)
        self.WHiddenOutput = generator.normal(
            loc=0.0,
            scale=0.1,
            size=(self.hiddenUnitCount, self.outputSize),
        )
        self.bOutput = np.zeros((1, self.outputSize), dtype=np.float64)

        self.trainingErrors = []
        self.epochCount = 0

    def OneHotEncode(self, y):
        yFlat = np.asarray(y).reshape(-1).astype(int)
        return np.eye(self.outputSize, dtype=np.float64)[yFlat]

    def Forward(self, X):
        netHidden = X @ self.WInputHidden + self.bHidden
        hiddenOutput = Sigmoid(netHidden)

        netOutput = hiddenOutput @ self.WHiddenOutput + self.bOutput
        output = Sigmoid(netOutput)

        return netHidden, hiddenOutput, netOutput, output

    def Fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        yOneHot = self.OneHotEncode(y)
        sampleCount = X.shape[0]

        previousError = None

        for epochIndex in range(self.maxEpochs):
            netHidden, hiddenOutput, netOutput, output = self.Forward(X)

            outputError = yOneHot - output
            totalError = 0.5 * np.sum((outputError) ** 2)
            averageError = float(totalError / sampleCount)
            self.trainingErrors.append(averageError)

            deltaOutput = outputError * output * (1.0 - output)
            deltaHidden = (deltaOutput @ self.WHiddenOutput.T) * hiddenOutput * (1.0 - hiddenOutput)

            gradientHiddenOutput = hiddenOutput.T @ deltaOutput / sampleCount
            gradientBOutput = np.sum(deltaOutput, axis=0, keepdims=True) / sampleCount
            gradientInputHidden = X.T @ deltaHidden / sampleCount
            gradientBHidden = np.sum(deltaHidden, axis=0, keepdims=True) / sampleCount

            self.WHiddenOutput = self.WHiddenOutput + self.learningRate * gradientHiddenOutput
            self.bOutput = self.bOutput + self.learningRate * gradientBOutput
            self.WInputHidden = self.WInputHidden + self.learningRate * gradientInputHidden
            self.bHidden = self.bHidden + self.learningRate * gradientBHidden

            self.epochCount = epochIndex + 1

            if previousError is not None:
                if abs(previousError - averageError) < self.tolerance:
                    break
            previousError = averageError

        return self

    def PredictProbabilities(self, X):
        X = np.asarray(X, dtype=np.float64)
        _, _, _, output = self.Forward(X)
        return output

    def Predict(self, X):
        probabilities = self.PredictProbabilities(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions.reshape(-1, 1)
