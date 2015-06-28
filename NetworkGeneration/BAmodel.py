import networkx as nw
import numpy as np


G = nw.barabasi_albert_graph(100, 5)

A = nw.adjacency_matrix(G)
A = np.array(A.todense())

X = A * -0.2


for i in range(100):
	X[i,i] = 0.5 - sum(X[i,:])

sigma = np.linalg.inv(X)

samplePoint = np.random.multivariate_normal(np.zeros(100),sigma,1200)
cov = np.cov(samplePoint.T)

np.savez('BAmodel100.npz', cov=cov, A=A)