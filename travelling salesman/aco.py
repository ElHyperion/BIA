import json
import math
from random import sample
from numpy.random import choice
from collections import namedtuple
import matplotlib.pyplot as plt


##############################
# Data preparation
##############################

with open('cities.json') as f:
    cities_dict = json.load(f)

cities = []
city_count = 15  # City count limiter
City = namedtuple('City', ['x', 'y', 'name'])
for city in cities_dict:
    if city_count == 0:
        break
    cities.append(City(cities_dict[city][0], cities_dict[city][1], city))
    city_count -= 1


class Edge():
    def __init__(self, city1, city2):
        self.c1 = city1
        self.c2 = city2
        self._pheromones = 0.25
        self.distance = math.sqrt((cities[city1].x - cities[city2].x)**2
                                  + (cities[city1].y - cities[city2].y)**2)

    def get_coords(self):
        return ((cities[self.c1].x, -cities[self.c1].y),
                (cities[self.c2].x, -cities[self.c2].y))

    def get_cities(self):
        return (self.c1, self.c2)

    def get_colour(self):
        return (0, 0, 0, edge.pheromones)

    @property
    def pheromones(self):
        return self._pheromones

    def add_pheromones(self, inc):
        self._pheromones += inc
        if self._pheromones > 1:
            self._pheromones = 1

    def decay_pheromones(self):
        self._pheromones = self._pheromones * 0.99

    def __repr__(self):
        return '(' + str(self.c1) + ', ' + str(self.c2) + ')'

    def __iter__(self):
        yield self.get_coords()


# Find all edges and calculate their lengths
edges = []
for c1 in range(len(cities)):
    for c2 in range(len(cities)):
        if c1 != c2:
            edgeFound = False
            for edge in edges:
                if edge.get_cities() == (c2, c1):
                    edgeFound = True
                    break
            if not edgeFound:
                edges.append(Edge(c1, c2))


##############################
# Ant Colony Optimization
##############################

colony = 1000
city_count = len(cities)
best_length, best_path = None, []
alpha, beta = 4, 6


print('Running for %d cities and a colony of %d ants (alpha: %f, beta: %f)' %
      (city_count, colony, alpha, beta))


def get_edge(c11, c2):
    for edge in edges:
        if edge.get_cities() == (c11, c2) or edge.get_cities() == (c2, c11):
            return edge


def get_colours():
    return [(0.4, 0.5, 0.6, edge.pheromones) for edge in edges]


def get_path_length(path):
    length = 0.0
    for i in range(1, len(path)):
        edge = get_edge(path[i - 1], path[i])
        length += edge.distance
    return length


# Generate the first iteration
best_path = sample(range(1, city_count), city_count - 1)
best_path.insert(0, 0)
best_path.append(0)
best_length = get_path_length(best_path)


# Prepare a graph
fig, ax = plt.subplots(1)

# Draw cities
ax.plot([city.x for city in cities], [-city.y for city in cities], 'bo')
for city in cities:
    plt.text(city.x, -city.y + 5, city.name)

# Draw path lines
best_line, = ax.plot([], [], 'r-')
phero_lines = []
for e in edges:
    edge_coords = list(zip(*e.get_coords()))
    line, = ax.plot(edge_coords[0], edge_coords[1], 'b-')
    phero_lines.append(line)

fig.canvas.set_window_title('ACO algorithm for TSP')
plt.axis('off')
plt.show(block=False)


# Iterate over each ant in a colony
for a in range(colony):

    path = [0]
    cur_city = 0
    remaining = [x for x in range(1, len(cities))]

    # Iterate over every city
    while len(remaining) > 0:
        next_city = None
        probabilities = []

        # Select the next city from remaining cities on a path
        for c in remaining:
            phero_weight = pow(get_edge(cur_city, c).pheromones, alpha)
            path_weight = pow(1 / get_edge(cur_city, c).distance, beta)
            probabilities.append(phero_weight * path_weight)

        probabilities = [probability / sum(probabilities)
                         for probability in probabilities]
        next_city = choice(remaining, p=probabilities)
        path.append(next_city)
        remaining.remove(next_city)
        cur_city = next_city

    path.append(0)
    cur_length = get_path_length(path)

    # Pheromone decay
    for e in edges:
        e.decay_pheromones()

    # Update pheromones based on ants path
    for i in range(len(path) - 1):
        get_edge(path[i], path[i + 1]).add_pheromones(1 / cur_length)

    if best_length > cur_length:
        best_length = cur_length
        best_path = path
        print(f'Found a shorter path: {best_length}')

    # Print pheromones nicely
    # print([f'{edge.pheromones:.2f}' for edge in edges])

    # Update the best path
    best_line.set_data([cities[c].x for c in best_path],
                       [-cities[c].y for c in best_path])
    for i in range(len(phero_lines)):
        phero_lines[i].set_color(edges[i].get_colour())

    plt.draw()
    plt.pause(1e-17)
    plt.title('Ant %2d   Shortest path %d' % (a, best_length))


plt.show()
