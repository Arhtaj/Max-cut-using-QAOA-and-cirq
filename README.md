By Thomas Mikhail, 2020
####################################################################################
The max cut problem is quite an easy simple problem to conceptualize. Given a
graph of N nodes and M edges between them, how can we cut it into two sets
of nodes such that the total weight of the cut edges is maximum. People are
generally successful at solving the problem visually, perhaps only running into
problems calculating the sums of weights, but the same could not be said for
computers, in fact, it is considered one the most difficult combinatorial
optimization problems to solve, being NP-complete. The goal of this project was
to implement and explore using the quantum approximate optimization
algorithm to solve this problem, using cirq to simulate it on a classical
computer.
The quantum approximate algorithm is a variational circuit, meaning that it is
parameterized. This is done by combining the quantum circuit with a classical
one, where a set of operations is sent to the quantum circuit, then the results
are read by the classical cpu, which are analyzed, and based on the results,
changes the parameters of the circuit before running it again. This is
performed many times to optimize the circuit for the problem, hence the name.
Once these parameters have been optimized, in this case by comparing the
cost value of the outputs, the problem can be tackled, which is generally a
combinatorial problem. However, because the circuit is also non-deterministic,
and the parameters that are selected are meant to make a correct result more
likely, not inevitable, and as such, the circuit is run several times, and results
are compared by the classical computer to find the best answer. Despite there
being a general randomness to results, it is far more likely to find the correct answer,
or something closer to it than a purely random function.
#######################################################################################
