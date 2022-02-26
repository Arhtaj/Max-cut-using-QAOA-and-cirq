import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx


# Reads input
f = open("Raw.txt", "r")
data = []
for row in f:
    data.append([int(x) for x in row.split()])


# Convert to graph and create qubits
numNodes = len(data)
Qs = cirq.LineQubit.range(numNodes)
G = nx.Graph()

for x in range(numNodes):
    G.add_node(x)

for i in range(len(data)):
    for j in range(len(data[i])):
        print(data[i][j])
        if data[i][j]!=0:
            G.add_edge(
                i, j, weight=data[i][j]
            )

nx.draw_circular(G,node_color='blue', node_size=1000, with_labels=True)
plt.show()