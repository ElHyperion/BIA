import json
import math
from random import uniform as rand, choice, sample
from collections import namedtuple
import matplotlib.pyplot as plt


##############################
# Data preparation
##############################

with open('cities.json') as f:
    citiesDict = json.load(f)

cities = []
cityCnt = -1  # City count limiter
City = namedtuple('City', ['x', 'y', 'name'])
for city in citiesDict:
    if cityCnt == 0:
        break
    cities.append(City(citiesDict[city][0], citiesDict[city][1], city))
    cityCnt -= 1

distances = [[] for city in cities]
for c1 in range(len(distances)):
    for c2 in range(len(distances)):
        city1, city2 = cities[c1], cities[c2]
        if c1 != c2:
            distances[c1].append(
                math.sqrt((city2.x - city1.x)**2 + (city2.y - city1.y)**2))
        else:
            distances[c1].append(0)


##############################
# Genetic algorithm
##############################

generations = 200
population = 2000
fittestCnt = 100
mutationThreshold = 0.25

cityCnt = len(cities)
paths, lengths = [], []


print('Running for %d cities and %d generations of %d population size!' %
      (cityCnt, generations, population))


def calculateLength(path):
    length = 0.0
    for i in range(1, len(path)):
        length += distances[path[i - 1]][path[i]]
    return length


# Generate initial generation
for i in range(population):
    path = sample(range(1, cityCnt), cityCnt - 1)
    path.insert(0, 0)
    path.append(0)
    paths.append(path)
    lengths.append(calculateLength(path))


# Prepare a graph
fig, ax = plt.subplots()

# Draw cities
ax.plot([city.x for city in cities], [-city.y for city in cities], 'bo')
for city in cities:
    plt.text(city.x, -city.y + 5, city.name)

# Draw path lines
line, = ax.plot([], [], 'r-')
fig.canvas.set_window_title('Genetic algorithm for TSP')
plt.axis('off')
plt.show(block=False)


for gen in range(generations):

    # Cross breeding
    children = []
    for p1 in range(population):
        children.append([0] * (cityCnt + 1))
        child = children[p1]

        # Select a secondary parent
        p2 = choice([x for x in range(population) if x != p1])

        # Select a gene cutoff range
        cutoff1, cutoff2 = sample([x for x in range(1, cityCnt)], 2)
        if cutoff1 > cutoff2:
            cutoff1, cutoff2 = cutoff2, cutoff1

        # Inherit from within the cutoff range of parent 1
        for j in range(cutoff1, cutoff2 + 1):
            child[j] = paths[p1][j]

        # Inherit from the rest of parent 2
        cutoff = 0
        for i in range(cityCnt):
            if paths[p2][i] not in child:
                cutoff += 1
                if cutoff >= cutoff1 and cutoff < cutoff2:
                    cutoff = cutoff2 + 1
                child[cutoff] = paths[p2][i]

        # Mutation
        if rand(0, 1) < mutationThreshold:
            i1, i2 = sample([x for x in range(1, cityCnt)], 2)
            child[i1], child[i2] = child[i2], child[i1]

    # Add the children to the population, if they are fitter
    for i in range(len(children)):
        newLength = calculateLength(children[i])
        if newLength < lengths[i]:
            paths[i], lengths[i] = children[i], newLength

    # Find the fittest in a generation
    lengths, paths = zip(*sorted(zip(lengths, paths)))
    lengths, paths = list(lengths), list(paths)
    path = paths[0]
    line.set_data([cities[x].x for x in path],
                  [-cities[x].y for x in path])
    plt.draw()
    plt.pause(1e-17)
    plt.title('Generation %2d   Shortest path %d' % (gen, lengths[0]))
    print('Generation %2d: %f' % (gen, lengths[0]))


plt.show()
