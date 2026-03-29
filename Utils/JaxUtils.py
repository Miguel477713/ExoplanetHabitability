import jax


def AddIntercept(X):
    ones = jax.numpy.ones((X.shape[0], 1), dtype=jax.numpy.float64)
    return jax.numpy.concatenate([ones, X], axis=1)
