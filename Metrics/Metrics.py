import numpy as np
import matplotlib.pyplot as plt
import jax.numpy


def ConfusionMatrix(yTrue, yPred, classCount):
    confusionMatrix = jax.numpy.zeros((classCount, classCount), dtype=int)

    for i in range(yTrue.shape[0]):
        trueClass = int(yTrue[i, 0])
        predClass = int(yPred[i, 0])
        confusionMatrix = confusionMatrix.at[trueClass, predClass].add(1)

    return confusionMatrix


def ClassificationMetrics(confusionMatrix):
    classCount = confusionMatrix.shape[0]

    precision = []
    recall = []
    f1 = []

    for k in range(classCount):
        tp = float(confusionMatrix[k, k])
        fp = float(confusionMatrix[:, k].sum() - confusionMatrix[k, k])
        fn = float(confusionMatrix[k, :].sum() - confusionMatrix[k, k])

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1_k = 2.0 * prec * rec / (prec + rec + 1e-8)

        precision.append(float(prec))
        recall.append(float(rec))
        f1.append(float(f1_k))

    macroPrecision = float(sum(precision) / len(precision))
    macroRecall = float(sum(recall) / len(recall))
    macroF1 = float(sum(f1) / len(f1))

    accuracy = float(np.trace(np.asarray(confusionMatrix)) / (np.asarray(confusionMatrix).sum() + 1e-8))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macroPrecision,
        "macro_recall": macroRecall,
        "macro_f1": macroF1,
        "accuracy": accuracy,
    }


def SaveConfusionMatrixPlot(confusionMatrix, classLabels, outputPath, title):
    confusionMatrixNp = np.asarray(confusionMatrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusionMatrixNp)
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(classLabels)))
    ax.set_yticks(range(len(classLabels)))
    ax.set_xticklabels(classLabels)
    ax.set_yticklabels(classLabels)

    for i in range(confusionMatrixNp.shape[0]):
        for j in range(confusionMatrixNp.shape[1]):
            ax.text(j, i, str(confusionMatrixNp[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(outputPath, bbox_inches="tight")
    plt.close(fig)


def SavePerClassMetricsPlot(metrics, classLabels, outputPath, title):
    x = np.arange(len(classLabels))
    width = 0.25

    precision = np.asarray(metrics["precision"], dtype=np.float64)
    recall = np.asarray(metrics["recall"], dtype=np.float64)
    f1 = np.asarray(metrics["f1"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(classLabels)
    ax.set_ylim(0.0, 1.05)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outputPath, bbox_inches="tight")
    plt.close(fig)


def PrintMetrics(metrics):
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("Macro Precision:", metrics["macro_precision"])
    print("Macro Recall:", metrics["macro_recall"])
    print("Macro F1:", metrics["macro_f1"])
