import numpy
import jax.numpy

from Models.LogisticRegression import LogisticRegressionCoefficients, Softmax
from Models.DecisionTree import DecisionTreeClassifierScratch
from Models.MLP import MLPClassifierScratch
from Utils.JaxUtils import AddIntercept


class MixtureModelClassifier:
    def __init__(
        self,
        classCount,
        gatingLearningRate=0.01,
        gatingIterations=2000,
        hiddenUnitCount=12,
        mlpLearningRate=0.05,
        mlpMaxEpochs=1000,
        mlpTolerance=1e-6,
        randomSeed=0,
    ):
        self.ClassCount = int(classCount)
        self.GatingLearningRate = float(gatingLearningRate)
        self.GatingIterations = int(gatingIterations)
        self.HiddenUnitCount = int(hiddenUnitCount)
        self.MlpLearningRate = float(mlpLearningRate)
        self.MlpMaxEpochs = int(mlpMaxEpochs)
        self.MlpTolerance = float(mlpTolerance)
        self.RandomSeed = int(randomSeed)

        self.FeatureMean = None
        self.FeatureStd = None
        self.LogisticCoefficients = None
        self.DecisionTree = None
        self.Mlp = None
        self.GatingCoefficients = None
        self.ExpertNames = ["LogisticExpert", "DecisionTreeExpert", "MlpExpert"]
        self.ExpertTrainingAccuracy = None

    def StandardizeWithTrainingStatistics(self, XTraining, XOther):
        XTraining = numpy.asarray(XTraining, dtype=numpy.float64)
        XOther = numpy.asarray(XOther, dtype=numpy.float64)

        mean = XTraining.mean(axis=0, keepdims=True)
        std = XTraining.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0

        XTrainingStandardized = (XTraining - mean) / std
        XOtherStandardized = (XOther - mean) / std
        return XTrainingStandardized, XOtherStandardized, mean, std

    def PredictLogisticProbabilities(self, XWithIntercept):
        logits = XWithIntercept @ self.LogisticCoefficients
        return numpy.asarray(Softmax(logits), dtype=numpy.float64)

    def PredictDecisionTreeProbabilities(self, XRaw):
        predictions = self.DecisionTree.Predict(XRaw).reshape(-1)
        probabilities = numpy.zeros((XRaw.shape[0], self.ClassCount), dtype=numpy.float64)
        probabilities[numpy.arange(XRaw.shape[0]), predictions.astype(int)] = 1.0
        return probabilities

    def PredictMlpProbabilities(self, XStandardized):
        return numpy.asarray(self.Mlp.PredictProbabilities(XStandardized), dtype=numpy.float64)

    def BuildExpertTargetLabels(self, YTraining, expertPredictions):
        sampleCount = YTraining.shape[0]
        expertCount = len(expertPredictions)
        expertCorrectness = numpy.zeros((sampleCount, expertCount), dtype=numpy.float64)

        for expertIndex, predictions in enumerate(expertPredictions):
            expertCorrectness[:, expertIndex] = (predictions.reshape(-1) == YTraining.reshape(-1)).astype(numpy.float64)

        expertTrainingAccuracy = expertCorrectness.mean(axis=0)
        targetLabels = numpy.argmax(expertCorrectness, axis=1).astype(int)

        fallbackExpert = int(numpy.argmax(expertTrainingAccuracy))
        noCorrectExpertMask = expertCorrectness.sum(axis=1) == 0.0
        targetLabels[noCorrectExpertMask] = fallbackExpert

        multipleCorrectMask = expertCorrectness.sum(axis=1) > 1.0
        if numpy.any(multipleCorrectMask):
            bestExpertOrder = numpy.argsort(-expertTrainingAccuracy)
            for sampleIndex in numpy.where(multipleCorrectMask)[0]:
                for expertIndex in bestExpertOrder:
                    if expertCorrectness[sampleIndex, expertIndex] > 0.5:
                        targetLabels[sampleIndex] = int(expertIndex)
                        break

        self.ExpertTrainingAccuracy = expertTrainingAccuracy
        return targetLabels

    def Fit(self, XTraining, YTraining):
        XTrainingRaw = numpy.asarray(XTraining, dtype=numpy.float64)
        YTrainingFlat = numpy.asarray(YTraining).reshape(-1).astype(int)

        XTrainingStandardized, _, self.FeatureMean, self.FeatureStd = self.StandardizeWithTrainingStatistics(XTrainingRaw, XTrainingRaw)
        XTrainingWithIntercept = AddIntercept(jax.numpy.asarray(XTrainingRaw))
        XTrainingWithIntercept = numpy.asarray(XTrainingWithIntercept, dtype=numpy.float64)

        self.LogisticCoefficients = numpy.asarray(
            LogisticRegressionCoefficients(
                jax.numpy.asarray(XTrainingWithIntercept),
                jax.numpy.asarray(YTrainingFlat).reshape(-1, 1),
                classCount=self.ClassCount,
                learningRate=self.GatingLearningRate,
                iterations=self.GatingIterations,
            ),
            dtype=numpy.float64,
        )

        self.DecisionTree = DecisionTreeClassifierScratch(
            classCount=self.ClassCount,
            maxDepth=8,
            minSamplesSplit=10,
            minGain=1e-6,
        )
        self.DecisionTree.Fit(XTrainingRaw, YTrainingFlat)

        self.Mlp = MLPClassifierScratch(
            inputSize=XTrainingStandardized.shape[1],
            hiddenUnitCount=self.HiddenUnitCount,
            outputSize=self.ClassCount,
            learningRate=self.MlpLearningRate,
            maxEpochs=self.MlpMaxEpochs,
            tolerance=self.MlpTolerance,
            randomSeed=self.RandomSeed,
        )
        self.Mlp.Fit(XTrainingStandardized, YTrainingFlat)

        logisticPredictions = numpy.argmax(self.PredictLogisticProbabilities(XTrainingWithIntercept), axis=1).reshape(-1, 1)
        decisionTreePredictions = self.DecisionTree.Predict(XTrainingRaw)
        mlpPredictions = self.Mlp.Predict(XTrainingStandardized)

        expertTargetLabels = self.BuildExpertTargetLabels(
            YTrainingFlat.reshape(-1, 1),
            [logisticPredictions, decisionTreePredictions, mlpPredictions],
        )

        self.GatingCoefficients = numpy.asarray(
            LogisticRegressionCoefficients(
                jax.numpy.asarray(XTrainingWithIntercept),
                jax.numpy.asarray(expertTargetLabels).reshape(-1, 1),
                classCount=len(self.ExpertNames),
                learningRate=self.GatingLearningRate,
                iterations=self.GatingIterations,
            ),
            dtype=numpy.float64,
        )

        return self

    def Predict(self, X):
        return self.PredictWithDetails(X)["Predictions"]

    def PredictWithDetails(self, X):
        XRaw = numpy.asarray(X, dtype=numpy.float64)
        XStandardized = (XRaw - self.FeatureMean) / self.FeatureStd
        XWithIntercept = AddIntercept(jax.numpy.asarray(XRaw))
        XWithIntercept = numpy.asarray(XWithIntercept, dtype=numpy.float64)

        logisticProbabilities = self.PredictLogisticProbabilities(XWithIntercept)
        decisionTreeProbabilities = self.PredictDecisionTreeProbabilities(XRaw)
        mlpProbabilities = self.PredictMlpProbabilities(XStandardized)

        gatingLogits = XWithIntercept @ self.GatingCoefficients
        gatingProbabilities = numpy.asarray(Softmax(gatingLogits), dtype=numpy.float64)

        combinedProbabilities = (
            gatingProbabilities[:, [0]] * logisticProbabilities
            + gatingProbabilities[:, [1]] * decisionTreeProbabilities
            + gatingProbabilities[:, [2]] * mlpProbabilities
        )

        predictions = numpy.argmax(combinedProbabilities, axis=1).reshape(-1, 1)

        return {
            "Predictions": predictions,
            "CombinedProbabilities": combinedProbabilities,
            "GatingProbabilities": gatingProbabilities, #expert: logistic
            "LogisticProbabilities": logisticProbabilities, #expert
            "DecisionTreeProbabilities": decisionTreeProbabilities, #expert
            "MlpProbabilities": mlpProbabilities, #expert
        }
