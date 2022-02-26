import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
print('Input P:')
p=int(input())

# Reads input
f = open("Raw.txt", "r")
data = []
for row in f:
    data.append([int(x) for x in row.split()])

# Convert to graph and create qubits
numNodes = len(data)
Qs = []
for i in range(numNodes):
    Qs.append(cirq.NamedQubit(str(i)))

G = nx.Graph()

for x in range(numNodes):
    G.add_node(x)

for i in range(len(data)):
    for j in range(len(data[i])):
        if data[i][j]!=0:
            G.add_edge(
                i, j, weight=data[i][j]
            )


# Symbols for the rotation angles in the QAOA circuit.
alpha = sympy.Symbol('alpha')
beta = sympy.Symbol('beta')

# Moments:
m1 = cirq.Moment(cirq.H.on_each(Qs))
m2 = (cirq.ZZ(Qs[u], Qs[v]) ** (alpha * w['weight']) for (u, v, w) in G.edges(data=True))
m3 = cirq.Moment(cirq.X(x) ** beta for x in Qs)
mout = (cirq.measure(x) for x in Qs)

mini=cirq.Circuit(m2,m3)

# Quantum Approximate Optimization Algorithm
qaoa = cirq.Circuit(m1).__add__(mini for i in range(p))
qaoa.append(mout)

print(qaoa)
alphaI = np.pi / 4
beta1 = np.pi / 2
sim = cirq.Simulator()

#Estimate cost
def estimate_cost(graph, samples):

    cost = 0.0

    for u, v, w in graph.edges(data=True):
        u_samples = samples[str(u)]
        v_samples = samples[str(v)]

        u_signs = (-1) ** u_samples
        v_signs = (-1) ** v_samples
        term_signs = u_signs * v_signs

        term_val = np.mean(term_signs) * w['weight']
        cost += term_val

    return -cost

#Make a cut
def Finalcut(S):
    coloring = []
    for node in G:
        if node in S:
            coloring.append('red')
        else:
            coloring.append('yellow')
    edges = G.edges(data=True)
    weights = [w['weight'] for (u,v, w) in edges]
    nx.draw_circular(
        G,
        node_color=coloring,
        node_size=1000,
        with_labels=True,
        width=weights)
    plt.show()
    size = nx.cut_size(G, S, weight='weight')

# Find Best parameters
alphaI = np.pi / 4
beta1 = np.pi / 2
sim = cirq.Simulator()
sample_results = sim.sample(
    qaoa,
    params={alpha: alphaI, beta: beta1},
    repetitions=20_000
)

grid_size = 5
exp_values = np.empty((grid_size, grid_size))
par_values = np.empty((grid_size, grid_size, 2))

for i, alphaI in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
    for j, beta1 in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
        samples = sim.sample(
            qaoa,
            params={alpha: alphaI, beta: beta1},
            repetitions=20000
        )
        exp_values[i][j] = estimate_cost(G, samples)
        par_values[i][j] = alphaI, beta1

best_exp_index = np.unravel_index(np.argmax(exp_values), exp_values.shape)
parameters = par_values[best_exp_index]



# Number of candidate cuts to compare
ncuts = 100
candidate_cuts = sim.sample(
    qaoa,
    params={alpha: parameters[0], beta: parameters[1]},
    repetitions=ncuts
)

# Variables to store best cut partitions and cut size.
SF = set()
TF = set()
FCsize = -np.inf

# Analyze each candidate cut.
for i in range(ncuts):
    candidate = candidate_cuts.iloc[i]
    ones = set(candidate[candidate==1].index)
    S = set()
    T = set()
    for node in G:
        if str(node) in ones:
            S.add(node)
        else:
            T.add(node)

    cut = nx.cut_size(
        G, S, T, weight='weight')
    #print(cut)
    if cut > FCsize:
        FCsize = cut
        SF = S
        TF = T

print('Final Cut:', FCsize)
Finalcut(SF)
