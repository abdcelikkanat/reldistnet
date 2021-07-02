import numpy as np


class SynNetGen:
    def __init__(self, N=None, init_z=None, A=None, tau=None, T=10, D=None, seed=None):

        self._N = N
        self._z = init_z
        self._A = A
        self._tau = tau
        self._T = T
        self._D = D
        self._seed = None

        # Check if the given parameters are valid or not

        if self._z is not None and self._N is not None:
            assert self._z.shape[0] == self._N, "The given number of nodes and initial latent positions are different!"
        if self._z is not None and self._D is not None:
            assert self._z.shape[1] == self._N, "The given dimension and initial latent sizes are different!"

        self._N = self._z.shape[0]
        self._D = self._z.shape[1]

        np.random.seed(self._seed)

        self._constructNetwork()

    def _performStep(self):

        print("---------")
        print( self._z[-self._tau:])
        next_z = np.sum(np.dot(self._z[-self._tau:], self._A), axis=1) + np.random.randn(self._N, self._D)

        return next_z

    def sampleEdge(self, lam, v, u):

        k = np.random.poisson(lam=lam, size=(1, ) )
        print(k)
        return 1 if k > 0 else 0


    def _constructNetwork(self):

        for _ in range(self._T):
            # Perform a step
            next_z = self._performStep()
            # Update the latent positions
            self._z = np.hstack((self._z, next_z))


if __name__ == '__main__':

    print(__file__)

    init_z = np.array([[-3, 0], [-2, -1], [-1, 0], [-2, 1], [1, 0], [2,-1], [3, 0], [2, 1]], dtype=np.float)
    A = np.diag([2, 3]).astype(dtype=np.float)

    sng = SynNetGen(N=None, init_z=init_z, A=A, tau=2, T=10, D=None)

    k = sng.sampleEdge(lam=2, u=0, v=1)

