import numpy as np
from sklearn.utils import check_random_state

class HoldOutClass:
    """
    Hold-out cross-validator generator. In the hold-out, the
    data is split only once into a train set and a test set.
    Unlike in other cross-validation schemes, the hold-out
    consists of only one iteration.

    Parameters
    ----------
    n : total number of samples
    test_size : 0 < float < 1
        Fraction of samples to use as test set. Must be a
        number between 0 and 1.
    random_state : int
        Seed for the random number generator.
    """
    def __init__(self, n, test_size=0.2, random_state=0):
        self.n = n
        self.test_size = test_size
        self.random_state = random_state

    def __iter__(self):
        n_test = int(np.ceil(self.test_size * self.n))
        n_train = self.n - n_test
        rng = check_random_state(self.random_state)
        permutation = rng.permutation(self.n)
        ind_test = permutation[:n_test]
        ind_train = permutation[n_test:n_test + n_train]
        yield ind_train, ind_test

def main():
    HoldOutCls = HoldOutClass(100,0.2)

    for h in HoldOutCls:
        (train_index,test_index) = h

    print(len(test_index))

if __name__ == "__main__":
    main()
