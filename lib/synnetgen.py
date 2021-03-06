import numpy as np
import pickle
import os


class SynNetGen:
    def __init__(self, init_z, init_beta=None, init_gamma=None, A=None, tau=None, T=10, seed=None):

        self._N = None  # number of nodes
        self._z = []  # list of latent locations
        self._edges = []  # list of edge lists
        self._beta = []  # list of beta values
        self._gamma = []  # list of gamma values
        self._A = A  # alpha, a list of DxD transition matrix
        self._tau = tau  # lag parameter
        self._T = T  # maximum time step
        self._D = None  # latent space dimension
        self._seed = seed  # seed value

        # Check if the given parameters are valid or not
        if type(init_z) is not list:
            init_z = [init_z]
        if type(init_z[0]) is not np.ndarray:
            raise ValueError("Initial locations must be Numpy array!")
        if init_z[0].ndim != 2:
            raise ValueError("Initial locations must be a 2-dimensional array!")


        # Set the initial locations and hyper-parameters
        self._z.extend(init_z)
        for _ in range(self._getCurrentTimeStep()):
            self._beta.append(init_beta)
            self._gamma.append(init_gamma)

        # Get the number of nodes and dimension size
        self._N = self._z[0].shape[0]
        self._D = self._z[0].shape[1]

        self._nodePairs = np.triu_indices(n=self._N, k=1)

        # Set the seed value
        np.random.seed(self._seed)

        # Sample the initial network
        for t in range(self._getCurrentTimeStep()):
            self._edges.append(self._sampleEdgesAt(t=t))

        self._constructNetwork()

    def _getCurrentTimeStep(self):

        return len(self._z)

    def _getLatentPositionsAt(self, t):

        if t > self._getCurrentTimeStep():
            raise ValueError("The given time must be smaller than the total number of steps!")

        return self._z[t]

    def _getLatentPositionOf(self, v, t):

        if v > self._N:
            raise ValueError("The given time must be smaller than the total number of steps!")

        z = self._getLatentPositionsAt(t)

        return z[v]

    def _computeNextLatentPositions(self):

        next_z = np.zeros_like(self._getLatentPositionsAt(0))
        lastTimeStep = self._getCurrentTimeStep()
        startingTimeStep = max(0, lastTimeStep - self._tau)
        for t in range(startingTimeStep, lastTimeStep):
            next_z += np.dot(self._getLatentPositionsAt(t), self._A[t-startingTimeStep])

        return next_z

    def _computeNextBeta(self):

        return self._beta[-1]

    def _computeNextGamma(self):

        return self._gamma[-1]

    def _computeDistance(self, v, u, t):

        z_v = self._getLatentPositionOf(v, t)
        z_u = self._getLatentPositionOf(u, t)

        return np.linalg.norm(z_v - z_u, ord=2)

    def _computeAllPairwiseDistanceAt(self, t):

        # An upper triangular matrix to store the distances
        distances = np.zeros(shape=(self._N, self._N), dtype=np.float)

        # Get the latent positions at time t
        z = self._getLatentPositionsAt(t)

        # Compute the pairwise distances
        for v, u in zip(self._nodePairs[0], self._nodePairs[1]):
            distances[v, u] = np.linalg.norm(z[v] - z[u], ord=2)

        return distances

    def _sampleEdge(self, lam, v, u):

        k = np.random.poisson(lam=lam, size=(1, ) )
        return 1 if k > 0 else 0

    def _sampleEdgesAt(self, t):

        # A list to store the edges
        edges = []

        # Compute lambda values
        lamValues = self._computeAllLamdaValuesAt(t=t)

        # Sample edges
        for v, u in zip(self._nodePairs[0], self._nodePairs[1]):
            if self._sampleEdge(lam=lamValues[v, u], v=v, u=u):
                edges.append([v, u])

        return edges

    def _computeAllLamdaValuesAt(self, t):

        # An upper triangular matrix to store the lambda values
        lamValues = np.zeros(shape=(self._N, self._N), dtype=np.float)

        # Compute all the pairwise distances
        distances = self._computeAllPairwiseDistanceAt(t)

        # Compute the lambda values
        for v, u in zip(self._nodePairs[0], self._nodePairs[1]):
            lamValues[v, u] = np.exp( self._beta[t] + self._gamma[t] - distances[v, u] )

        return lamValues

    def _constructNetwork(self):

        for currentTime in range(1, self._T):

            # Compute the next latent positions
            next_z = self._computeNextLatentPositions()
            next_beta = self._computeNextBeta()
            next_gamma = self._computeNextGamma()

            # Update the latent positions and hyper-parameters
            self._z.append(next_z)
            self._beta.append(next_beta)
            self._gamma.append(next_gamma)

            # Sample a graph for the next time step
            edges = self._sampleEdgesAt(t=currentTime)
            # Update list of edge lists
            self._edges.append(edges)

    def getEdgesAt(self, t):

        return self._edges[t]

    def getAllEdges(self):

        return self._edges

    def saveGraph(self, filePath, format='edgelist'):

        if format == 'edgelist':

            with open(filePath, "wb") as fp:
                pickle.dump({'edges': self._edges, 'z': self._z, 'beta': self._beta, 'gamma': self._gamma}, fp)

        else:

            raise ValueError("Invalid file format!")

    def animate(self, filepath=None):

        assert self._D == 2, "The dimension size must be 2!"

        # import the required packages
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        nodelist = [v for v in range(self._N)]
        edgelist = self._edges[0]
        g = nx.Graph()
        g.add_nodes_from(nodelist)
        g.add_edges_from(edgelist)

        # Set the artistic features
        pos = nx.spring_layout(g, seed=self._seed)  # positions for all nodes
        background_color = 'white'
        node_options = {"node_size": 200, "alpha": 0.25, "node_color": "gray", "linewidths": 0.85, "edgecolors": "k"}
        node_label_options = {"font_size": 8, "font_color": 'k', "font_weight": 'normal'}
        edge_options = {"width": 1, "alpha": 0.125, "edge_color": 'gray', "arrows": False}

        # Determine the axis limits for the latent position figure
        margin = 0.1
        xLeftLimit1, yBelowLimit1 = np.min(np.min(self._z)), np.min(np.min(self._z)) #np.min(np.min(self._z, axis=1), axis=0)
        xRightLimit1, yTopLimit1 = np.max(np.max(self._z)), np.max(np.max(self._z)) #np.max(np.max(self._z, axis=1), axis=0)

        # Determine the axis limits for the graph figure
        xLeftLimit2, yBelowLimit2 = np.min(list(pos.values()), axis=0)
        xRightLimit2, yTopLimit2 = np.max(list(pos.values()), axis=0)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.set_facecolor(background_color)

        def init():
            # Reset
            ax[0].clear()
            ax[0].set_facecolor(background_color)
            # Plot the latent positions
            ax[0].plot(self._z[0][:, 0], self._z[0][:, 1], 'r.', alpha=0.75)

            # Reset
            ax[1].clear()
            ax[1].set_facecolor(background_color)
            # Set the plot limits
            ax[0].set_xlim(xLeftLimit1 - margin, xRightLimit1 + margin)
            ax[0].set_ylim(yBelowLimit1 - margin, yTopLimit1 + margin)
            ax[1].set_xlim(xLeftLimit2 - margin, xRightLimit2 + margin)
            ax[1].set_ylim(yBelowLimit2 - margin, yTopLimit2 + margin)

            # draw nodes
            nx.draw_networkx_nodes(g, pos, ax=ax[1], nodelist=nodelist, **node_options)
            nx.draw_networkx_labels(g, pos, ax=ax[1], labels={node: str(node) for node in nodelist}, **node_label_options)
            # draw edges
            nx.draw_networkx_edges(g, pos, ax=ax[1], edgelist=self._edges[0], **edge_options)

        def animate(t):

            # Reset
            ax[0].clear()
            ax[0].set_facecolor(background_color)
            # Plot the latent positions
            ax[0].plot(self._z[t][:, 0], self._z[t][:, 1], 'r.', alpha=0.75)

            ax[0].title.set_text('Latent Space')

            # Reset
            ax[1].clear()
            ax[1].set_facecolor(background_color)
            # Set the plot limits
            ax[0].set_xlim(xLeftLimit1 - margin, xRightLimit1 + margin)
            ax[0].set_ylim(yBelowLimit1 - margin, yTopLimit1 + margin)
            ax[1].set_xlim(xLeftLimit2 - margin, xRightLimit2 + margin)
            ax[1].set_ylim(yBelowLimit2 - margin, yTopLimit2 + margin)

            # draw nodes
            nx.draw_networkx_nodes(g, pos, ax=ax[1], nodelist=nodelist, **node_options)
            nx.draw_networkx_labels(g, pos, ax=ax[1], labels={node: str(node) for node in nodelist}, **node_label_options)
            # draw edges
            nx.draw_networkx_edges(g, pos, ax=ax[1], edgelist=self._edges[t], **edge_options)

            ax[1].get_xaxis().set_ticks([])
            ax[1].get_yaxis().set_ticks([])
            ax[1].title.set_text('Graph Space')
            ax[1].set_xlabel("Number of edges: {}".format(len(self._edges[t])))
            # Remove the previous time text and add the new one
            for txt in fig.texts:
                txt.remove()
            fig.text(0.5, 0.95, "Time: {}".format(t), horizontalalignment='center', fontsize=12)

        anim = FuncAnimation(fig, animate, init_func=init, frames=self._T, interval=1000, )

        if filepath is not None:
            anim.save(filepath, writer='imagemagick', savefig_kwargs={"facecolor": background_color})

        return anim

if __name__ == '__main__':

    print(__file__)

    init_z = np.array([[-3, 0], [-2, -1], [-1, 0], [-2, 1], [1, 0], [2,-1], [3, 0], [2, 1]], dtype=np.float)
    A = np.diag([1.0001, 1.0003]).astype(dtype=np.float)
    A = [A, A, A]
    beta = 2
    gamma = 4
    tau = 3
    T = 10
    sng = SynNetGen(init_z=init_z, init_beta=beta, init_gamma=gamma, A=A, tau=tau, T=T)

    filename = "graph_n={}_d={}_beta={}_gamma={}_tau={}_T={}.pkl".format(init_z.shape[0], init_z.shape[1], beta, gamma, tau, T)
    sng.saveGraph(filePath=os.path.join('./networks/', filename))

    edges = sng.getEdgesAt(t=0)
    sng.animate(filepath='./anim.gif')
    #k = sng._constructNetwork(lam=2, u=0, v=1)

