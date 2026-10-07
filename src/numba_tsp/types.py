import numpy as np

type DistMatrix[n: int] = np.ndarray[tuple[n, n], np.dtype[np.floating]]
type IntArray[n: int] = np.ndarray[tuple[n], np.dtype[np.signedinteger]]
