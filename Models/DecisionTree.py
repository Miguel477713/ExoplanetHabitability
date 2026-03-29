import numpy as np


class DecisionTreeNode:
    def __init__(
        self,
        *,
        isLeaf,
        predictedClass,
        classCounts,
        featureIndex=None,
        threshold=None,
        left=None,
        right=None,
        depth=0,
        sampleCount=0,
        impurity=0.0,
        informationGain=0.0,
    ):
        self.isLeaf = isLeaf
        self.predictedClass = int(predictedClass)
        self.classCounts = np.asarray(classCounts, dtype=int)
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = int(depth)
        self.sampleCount = int(sampleCount)
        self.impurity = float(impurity)
        self.informationGain = float(informationGain)


def EntropyFromCounts(classCounts):
    total = int(np.sum(classCounts))
    if total == 0:
        return 0.0

    probabilities = classCounts / total
    nonZeroProbabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(nonZeroProbabilities * np.log2(nonZeroProbabilities)))


def MajorityClass(y, classCount):
    counts = np.bincount(y.astype(int), minlength=classCount)
    return int(np.argmax(counts)), counts


def CandidateThresholds(featureValues):
    distinctValues = np.unique(featureValues)
    if distinctValues.shape[0] <= 1:
        return np.array([], dtype=np.float64)
    return (distinctValues[:-1] + distinctValues[1:]) / 2.0


def BestSplit(X, y, classCount):
    sampleCount, featureCount = X.shape
    _, parentCounts = MajorityClass(y, classCount)
    parentImpurity = EntropyFromCounts(parentCounts)

    bestFeatureIndex = None
    bestThreshold = None
    bestGain = -np.inf
    bestLeftMask = None

    for featureIndex in range(featureCount):
        thresholds = CandidateThresholds(X[:, featureIndex])

        for threshold in thresholds:
            leftMask = X[:, featureIndex] <= threshold
            rightMask = ~leftMask

            leftCount = int(np.sum(leftMask))
            rightCount = int(np.sum(rightMask))

            if leftCount == 0 or rightCount == 0:
                continue

            leftCounts = np.bincount(y[leftMask].astype(int), minlength=classCount)
            rightCounts = np.bincount(y[rightMask].astype(int), minlength=classCount)

            leftImpurity = EntropyFromCounts(leftCounts)
            rightImpurity = EntropyFromCounts(rightCounts)

            weightedChildImpurity = (
                (leftCount / sampleCount) * leftImpurity
                + (rightCount / sampleCount) * rightImpurity
            )
            informationGain = parentImpurity - weightedChildImpurity

            if informationGain > bestGain:
                bestGain = informationGain
                bestFeatureIndex = featureIndex
                bestThreshold = float(threshold)
                bestLeftMask = leftMask

    return bestFeatureIndex, bestThreshold, float(bestGain), bestLeftMask, parentImpurity, parentCounts


class DecisionTreeClassifierScratch:
    def __init__(self, classCount, maxDepth=8, minSamplesSplit=10, minGain=1e-6):
        self.classCount = int(classCount)
        self.maxDepth = int(maxDepth)
        self.minSamplesSplit = int(minSamplesSplit)
        self.minGain = float(minGain)
        self.root = None

    def Fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1).astype(int)
        self.root = self.GrowTree(X, y, depth=0)
        return self

    def GrowTree(self, X, y, depth):
        predictedClass, classCounts = MajorityClass(y, self.classCount)
        nodeImpurity = EntropyFromCounts(classCounts)
        sampleCount = X.shape[0]

        isPure = np.count_nonzero(classCounts) == 1
        reachedMaxDepth = depth >= self.maxDepth
        tooSmallToSplit = sampleCount < self.minSamplesSplit

        if isPure or reachedMaxDepth or tooSmallToSplit:
            return DecisionTreeNode(
                isLeaf=True,
                predictedClass=predictedClass,
                classCounts=classCounts,
                depth=depth,
                sampleCount=sampleCount,
                impurity=nodeImpurity,
            )

        featureIndex, threshold, informationGain, leftMask, _, _ = BestSplit(X, y, self.classCount)

        if featureIndex is None or informationGain < self.minGain:
            return DecisionTreeNode(
                isLeaf=True,
                predictedClass=predictedClass,
                classCounts=classCounts,
                depth=depth,
                sampleCount=sampleCount,
                impurity=nodeImpurity,
            )

        leftNode = self.GrowTree(X[leftMask], y[leftMask], depth + 1)
        rightNode = self.GrowTree(X[~leftMask], y[~leftMask], depth + 1)

        return DecisionTreeNode(
            isLeaf=False,
            predictedClass=predictedClass,
            classCounts=classCounts,
            featureIndex=featureIndex,
            threshold=threshold,
            left=leftNode,
            right=rightNode,
            depth=depth,
            sampleCount=sampleCount,
            impurity=nodeImpurity,
            informationGain=informationGain,
        )

    def PredictOne(self, x, node):
        if node.isLeaf:
            return node.predictedClass

        if x[node.featureIndex] <= node.threshold:
            return self.PredictOne(x, node.left)
        return self.PredictOne(x, node.right)

    def Predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        predictions = [self.PredictOne(row, self.root) for row in X]
        return np.asarray(predictions, dtype=int).reshape(-1, 1)

    def TreeDepth(self, node):
        if node is None:
            return 0
        if node.isLeaf:
            return 1
        return 1 + max(self.TreeDepth(node.left), self.TreeDepth(node.right))

    def Depth(self):
        return self.TreeDepth(self.root)

    def _LeafCount(self, node):
        if node is None:
            return 0
        if node.isLeaf:
            return 1
        return self._LeafCount(node.left) + self._LeafCount(node.right)

    def LeafCount(self):
        return self._LeafCount(self.root)


def TreeToRules(node, featureNames, indent=""):
    if node.isLeaf:
        countsText = ", ".join([f"class_{i}={int(count)}" for i, count in enumerate(node.classCounts)])
        return (
            f"{indent}Predict class {node.predictedClass} "
            f"[samples={node.sampleCount}, impurity={node.impurity:.6f}, {countsText}]\n"
        )

    featureName = featureNames[node.featureIndex]
    text = (
        f"{indent}IF {featureName} <= {node.threshold:.6f} "
        f"[samples={node.sampleCount}, impurity={node.impurity:.6f}, gain={node.informationGain:.6f}]\n"
    )
    text += TreeToRules(node.left, featureNames, indent + "    ")
    text += f"{indent}ELSE\n"
    text += TreeToRules(node.right, featureNames, indent + "    ")
    return text
