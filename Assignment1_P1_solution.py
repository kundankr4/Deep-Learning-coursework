import numpy as np

class MU:
    __data: np.ndarray
        
    @property
    def data(self) -> np.ndarray:
        return self.__data
        
    @data.setter
    def data(self, data: list[float]) -> None:
        self.__data = np.array(data)

    def __init__(self, data: list[float]):
        self.data = data

    def estimate_mean(self) -> float:
        return float(np.mean(self.data))

"""
Write your class here, no need to change the name of class and function.
In the constructor, read data from given .txt file,  use numpy.loadtxt.
"""
class MLE(MU):
    def __init__(self, filename: str="data1.txt") -> None:
        data: list[float] = np.loadtxt(filename).tolist()
        super().__init__(data)

    def estimate_variance(self) -> float:
        return float(np.var(self.data))

    """
    # OR:
    def estimate_variance(self) -> float:
        mean = self.mean_estimator()
        return float(np.mean((self.data - mean) ** 2))
    """

"""
Call class MLE and estimator mean and variance
"""

mle = MLE()
mu = mle.estimate_mean()
sigma = mle.estimate_variance()

print(mu)
print(sigma)
