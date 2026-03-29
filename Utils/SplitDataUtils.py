import jax


def StratifiedKFoldIndices(Y, k=5, seed=0):
    key = jax.random.PRNGKey(seed)

    yFlat = jax.numpy.ravel(Y).astype(int)
    classes = jax.numpy.unique(yFlat)

    perFoldTestIndices = [[] for _ in range(k)]

    for classValue in classes:
        classIndices = jax.numpy.where(yFlat == classValue)[0]

        key, subkey = jax.random.split(key)
        classIndices = jax.random.permutation(subkey, classIndices)

        classCount = int(classIndices.shape[0])

        for foldIndex in range(k):
            start = (foldIndex * classCount) // k
            end = ((foldIndex + 1) * classCount) // k
            perFoldTestIndices[foldIndex].append(classIndices[start:end])

    folds = []
    allIndices = jax.numpy.arange(Y.shape[0])

    for foldIndex in range(k):
        testIndices = jax.numpy.concatenate(perFoldTestIndices[foldIndex], axis=0)

        mask = jax.numpy.ones(allIndices.shape[0], dtype=bool)
        mask = mask.at[testIndices].set(False)
        trainIndices = allIndices[mask]

        folds.append((trainIndices, testIndices))

    return folds


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
