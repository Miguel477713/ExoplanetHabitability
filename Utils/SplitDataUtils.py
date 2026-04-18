import jax


def StratifiedTrainTestIndices(Y, trainFraction=0.8, seed=0):
    key = jax.random.PRNGKey(seed)

    yFlat = jax.numpy.ravel(Y).astype(int)
    classes = jax.numpy.unique(yFlat)

    trainIndices = []
    testIndices = []

    for classValue in classes:
        classIndices = jax.numpy.where(yFlat == classValue)[0]

        key, subkey = jax.random.split(key)
        shuffledClassIndices = jax.random.permutation(subkey, classIndices)

        classCount = shuffledClassIndices.shape[0]
        classTrainCount = int(trainFraction * classCount)

        if classCount > 1:
            classTrainCount = max(1, min(classCount - 1, classTrainCount))

        trainIndices.append(shuffledClassIndices[:classTrainCount])
        testIndices.append(shuffledClassIndices[classTrainCount:])

    trainIndices = jax.numpy.concatenate(trainIndices)
    testIndices = jax.numpy.concatenate(testIndices)

    key, subkey = jax.random.split(key)
    trainIndices = jax.random.permutation(subkey, trainIndices)

    key, subkey = jax.random.split(key)
    testIndices = jax.random.permutation(subkey, testIndices)

    return trainIndices, testIndices


def PrintClassDistribution(name, Y):
    yFlat = jax.numpy.ravel(Y).astype(int)
    classes, counts = jax.numpy.unique(yFlat, return_counts=True)

    print(name)
    total = int(counts.sum())
    for classValue, count in zip(classes.tolist(), counts.tolist()):
        print(f"class {classValue}: {count} ({count / total:.4f})")
