# Biologically Inspired Algorithms
Repository for some of my solved tasks for a college course titled Biologically Inspired Algorithms.

The function extremes directory contains test function definitions in **functions.py** and some basic global extreme-searching algorithms defined in **main.py**, or algorithms using swarm logic defined in **swarms.py**.

The travelling salesman folder contains point definitions in **cities.json** and two algorithms that use them.

### Algorithms for searching global extremes:
* Blind search
* Hill climb
* Simulated annealing
* PSO (Particle Swarm Optimisation)
* Firefly swarm
* Differential

### Algorithms for solving TSP:
* Genetic
* ACO (Ant Colony Optimisation)

## Usage and requirements
Python 3.7 or newer, Matplotlib

### Global extreme
Uncomment one of the bottom *runFunction* lines in main.py or swarms.py. The first argument is the test function, second is a tuple of the function limits, and the third is the algorithm class to use in solving it.

### TSP
Open either **genetic.py** or **aco.py** and run it. Optionally, you can limit the city count (to make the algorithm run faster) by changing the cityCnt variable to a non-negative number.

## Screenshots

### Function extremes
![Function extremes screenshot2](function%20extremes/screenshot2.png)
![Function extremes screenshot1](function%20extremes/screenshot1.png)

### Travelling salesman problem
![Travelling salesman screenshot](travelling%20salesman/screenshot.png)
