import numpy as np


def multisample_collate(x):
    num_return_types = len(x[0])
    return (*[np.stack([y[i] for y in x]) for i in range(num_return_types)],)
