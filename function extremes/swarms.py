import functions as fnc
from random import uniform as rand, sample
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import numpy as np
from collections import namedtuple
from copy import copy


p_count = 100


class AlgorithmGeneric:
    best_result = None

    def __init__(self, function, limits):
        self.function = function
        self.limits = limits
        self.best_id = None
        self.run()

    def generate_initial(self, particle_class):
        self.population = [particle_class(rand(self.limits[0], self.limits[1]),
                                          rand(self.limits[0], self.limits[1]))
                           for i in range(p_count)]


class Particle:
    particle_count = 0
    Point = namedtuple('Point', ['x', 'y', 'result', 'id', 'isBest'])

    def __init__(self, x, y, id=None):
        self.result = None
        self.x = x
        self.y = y
        self.vector = (0, 0)
        if id is not None:
            self.id = id
        else:
            self.id = Particle.particle_count
            Particle.particle_count += 1


class ParticlePSO(Particle):
    def __init__(self, x, y):
        Particle.__init__(self, x, y)
        self.best_xy = (0, 0)
        self.best = None


class Firefly(Particle):
    Point = namedtuple('Point', ['x', 'y', 'result', 'id', 'intensity', 'isBest'])

    def __init__(self, x, y):
        Particle.__init__(self, x, y)
        self.intensity = None


class SOMA(AlgorithmGeneric):
    pass


class PSO(AlgorithmGeneric):
    def run(self):

        # Generate initial population
        self.generate_initial(ParticlePSO)

        # Personal best and global best coefficients
        p_best, g_best = 0.02, 0.05
        wgt = 1.2

        while True:
            wgt += 0.01

            # Update results and find the best particle
            for i, p in enumerate(self.population):
                p.result = self.function((p.x, p.y))
                if self.best_result is None or self.best_result > p.result:
                    self.best_result = p.result
                    self.best_id = p.id
                    print('Found a new best solution! ' + str(self.best_result))
                self.population[i] = p

                # Update personal best
                if p.best is None or p.best > p.result:
                    p.best = p.result
                    p.best_xy = (p.x, p.y)

            # Iterate over particles
            for i, p in enumerate(self.population):
                p.result = self.function((p.x, p.y))

                # Slow down the best particle
                if i == self.best_id:
                    p.vector = (p.vector[0] / 10, p.vector[1] / 10)
                else:
                    best = self.population[self.best_id]
                    p.vector = (p.vector[0] / wgt + (best.x - p.x) * g_best
                                + (p.best_xy[0] - p.x) * p_best,
                                p.vector[1] / wgt + (best.y - p.y) * g_best
                                + (p.best_xy[1] - p.y) * p_best)

                p.x = max(self.limits[0], min(self.limits[1], p.x + p.vector[0]))
                p.y = max(self.limits[0], min(self.limits[1], p.y + p.vector[1]))
                p.result = self.function((p.x, p.y))

                self.population[i] = p
                yield Particle.Point(p.x, p.y, p.result, p.id, i == self.best_id)


class Differential(AlgorithmGeneric):
    def run(self):

        # Generate initial population with results
        self.generate_initial(Particle)
        for p in self.population:
            p.result = self.function((p.x, p.y))

        # Migration count, crossover rate and scaling factor
        mig = 1000
        CR = 0.5
        F = 0.5

        # Iterate over migrations
        for cur_mig in range(mig):
            new_population = []

            # Iterate over population
            for i, p in enumerate(self.population):

                # Find the best particle
                if self.best_result is None or self.best_result > p.result:
                    self.best_result = p.result
                    self.best_id = p.id
                    print(f'Found a new best solution (mig {cur_mig}/{mig}): {self.best_result}')

                # Mutation operation
                tgts = sample(self.population, 3)
                x = max(self.limits[0], min(self.limits[1],
                                            tgts[0].x + F * (tgts[1].x - tgts[2].x)))

                y = max(self.limits[0], min(self.limits[1],
                                            tgts[0].y + F * (tgts[1].y - tgts[2].y)))

                # Crossover operation
                add_x, add_y = False, False
                if rand(0, 1) < CR:
                    add_x = True
                if rand(0, 1) < CR:
                    add_y = True
                if add_x or add_y:
                    new_result = self.function((x, y))

                    # Create a new particle, if better than the current one
                    if p.result > new_result:
                        new_p = copy(p)
                        new_p.x = x if add_x else new_p.x
                        new_p.y = y if add_y else new_p.y
                        new_p.result = new_result
                        new_population.append(new_p)

            # Update old population with new one
            for p in new_population:
                self.population[p.id] = p

            for p in self.population:
                yield Particle.Point(p.x, p.y, p.result, p.id, p.id == self.best_id)


class FireflySwarm(AlgorithmGeneric):
    def run(self):

        # Generate initial population
        self.generate_initial(Firefly)

        # Initial weight and gamma
        wgt = 1.2
        gamma = 0.1

        while True:
            # wgt += 0.02

            # Update results and intensity and find the best firefly
            for i, p in enumerate(self.population):
                p.result = self.function((p.x, p.y))
                p.intensity = 1 / p.result
                if self.best_result is None or self.best_result > p.result:
                    self.best_result = p.result
                    self.best_id = p.id
                    print('Found a new best solution! ' + str(self.best_result))
                self.population[i] = p

            # Iterate over all fireflies
            for i, p in enumerate(self.population):

                # Find a more attractive firefly and calculate relative intensity
                for tgt in self.population:
                    if tgt.id != p.id:
                        distance = np.sqrt(pow(p.x - tgt.x, 2)
                                           + pow(p.y - tgt.y, 2))
                        exponent = np.exp(-gamma * pow(distance, 2))
                        rel_intensity = tgt.intensity * exponent
                        if p.intensity < rel_intensity:
                            p.vector = (p.vector[0] + (tgt.x - p.x) / 5,
                                        p.vector[1] + (tgt.y - p.y) / 5)
                            break

                # Update vectors
                p.vector = (p.vector[0] / wgt,
                            p.vector[1] / wgt)
                _x = max(self.limits[0], min(
                         self.limits[1], p.x + p.vector[0]))
                _y = max(self.limits[0], min(
                         self.limits[1], p.y + p.vector[1]))
                p.x = _x
                p.y = _y
                p.vector = (p.vector[0] / wgt, p.vector[1] / wgt)
                p.result = self.function((p.x, p.y))

                self.population[i] = p
                yield Firefly.Point(p.x, p.y, p.result, p.id,
                                    p.intensity, i == self.best_id)


def runFunction(function, limits, algorithm):
    print(f'Running {algorithm.__name__} on {function.__name__.capitalize()}...')

    pointsX = [0] * p_count
    pointsY = [0] * p_count
    pointsZ = [0] * p_count
    sizes = [10] * p_count
    colours = [(1, 1, 0)] * p_count
    bestPoint = None

    N = 80
    X = np.linspace(limits[0], limits[1], N)
    Y = np.linspace(limits[0], limits[1], N)
    X, Y = np.meshgrid(X, Y)
    Z = function((X, Y))

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.w_xaxis.set_pane_color((0.0, 0.1, 0.2, 0.8))
    ax.w_yaxis.set_pane_color((0.0, 0.1, 0.2, 0.8))
    ax.w_zaxis.set_pane_color((0.0, 0.1, 0.2, 0.8))
    ax.plot_surface(X, Y, Z, cmap=cm.bone_r, linewidth=0,
                    alpha=0.3, antialiased=True)
    sc = ax.scatter(pointsX, pointsY, pointsZ,
                    c=colours, marker='o', s=sizes, alpha=.8)
    scBest = ax.scatter([], [], [], c='r', marker='+', s=1000)

    plt.ylabel(function.__name__.capitalize())
    plt.title(function.__name__.capitalize()
              + ' (' + algorithm.__name__ + ')')
    plt.show(block=False)

    for result in algorithm(function, limits).run():
        if result.isBest:
            bestPoint = (result.x, result.y, result.result)
        try:
            pointsX[result.id] = result.x
            pointsY[result.id] = result.y
            pointsZ[result.id] = result.result
        except IndexError:
            print(result.id)
            print('Algorithm finished!')
            plt.close()
            break

        try:  # Firefly
            sizes[result.id] = min(pow(result.intensity * 20, 4), 80)
            colours[result.id] = (1, 1, 1)
        except AttributeError:
            sizes[result.id] = 10
        if result.id == p_count - 1:
            sc._offsets3d = (pointsX, pointsY, pointsZ)
            sc.set_sizes(sizes)
            sc.set_color(colours)
            scBest._offsets3d = (
                [bestPoint[0]], [bestPoint[1]], [bestPoint[2]])
            plt.draw()
            plt.pause(1e-17)


# runFunction(fnc.sphere, (-5.12, 5.12), PSO)
# runFunction(fnc.griewank, (-10, 10), PSO)
# runFunction(fnc.rosenbrock, (-10, 10), FireflySwarm)
runFunction(fnc.rastrigin, (-5.12, 5.12), Differential)
# runFunction(fnc.rastrigin, (-5.12, 5.12), FireflySwarm)
# runFunction(fnc.schwefel, (-500, 500), FireflySwarm)
# runFunction(fnc.levy, (-10, 10), FireflySwarm)
# runFunction(fnc.zakharov, (-10, 10), FireflySwarm)
# runFunction(fnc.michalewicz, (0, np.pi), FireflySwarm)

plt.show()
