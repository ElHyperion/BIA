import functions as fnc
from random import uniform as rand
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


iterations = 1000


class AlgorithmGeneric:
    bestResult = None

    def __init__(self, function, limits):
        self.function = function
        self.limits = limits
        self.run()


class BlindSearch(AlgorithmGeneric):
    def run(self):
        for iter in range(iterations):
            x = rand(self.limits[0], self.limits[1])
            y = rand(self.limits[0], self.limits[1])
            result = {
                'x': x,
                'y': y,
                'value': self.function([x, y]),
            }
            if self.bestResult is None or self.bestResult > result['value']:
                self.bestResult = result['value']
                yield result


class HillClimb(AlgorithmGeneric):
    def run(self):
        scatter = (self.limits[1] - self.limits[0]) / 10
        points = 100
        lastPos = None

        while True:
            x = y = 0.0
            foundBetter = False
            for i in range(points):
                if lastPos is None:
                    x = y = rand(self.limits[0], self.limits[1])
                else:
                    x = lastPos[0] + rand(-scatter, scatter)
                    y = lastPos[1] + rand(-scatter, scatter)
                value = self.function([x, y])
                if self.bestResult is None or value < self.bestResult:
                    foundBetter = True
                    self.bestResult = value
                    lastPos = (x, y)
                    result = {
                        'x': x,
                        'y': y,
                        'value': value
                    }
                    yield result

            if not foundBetter:
                print('Climbed to the top!')
                break


class SimulatedAnnealing(AlgorithmGeneric):
    def run(self):
        scatter = (self.limits[1] - self.limits[0]) / 10
        temp = 500
        lastPos = None
        lastResult = 0.0

        while True:
            x = y = 0.0
            if lastPos is None:
                x = y = rand(self.limits[0], self.limits[1])
            else:
                x = lastPos[0] + rand(-scatter, scatter)
                y = lastPos[1] + rand(-scatter, scatter)
            lastResult = value = self.function([x, y])

            if self.bestResult is None or value < self.bestResult:
                print("found better")
                self.bestResult = value
                lastPos = (x, y)
                result = {
                    'x': x,
                    'y': y,
                    'value': value
                }
                yield result

            else:
                print('Looking for another solution...')
                while temp > 0.1:
                    x = y = rand(self.limits[0], self.limits[1])
                    value = self.function([x, y])
                    delta = value - lastResult
                    temp = temp * 0.99
                    if value < lastResult + np.exp(-delta / temp):
                        lastPos = (x, y)
                        if value < lastResult:
                            print('Found better, temp: ' + str(temp))
                        else:
                            print('Found worse, temp: ' + str(temp))
                        result = {
                            'x': x,
                            'y': y,
                            'value': value
                        }
                        yield result
                        break

            if temp <= 0.1:
                print('Ran out of temperature')
                break


def drawPlot(function, limits, results):
    N = 100
    X = np.linspace(limits[0], limits[1], N)
    Y = np.linspace(limits[0], limits[1], N)
    X, Y = np.meshgrid(X, Y)
    Z = function((X, Y))

    resX = [result['x'] for result in results]
    resY = [result['y'] for result in results]
    resZ = [result['value'] for result in results]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(resX, resY, resZ, c='g', marker='o')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.ylabel(function.__name__.capitalize())
    plt.title(function.__name__.capitalize())
    plt.show()


def runFunction(function, limits, algorithm):
    results = []

    for result in algorithm(function, limits).run():
        results.append(result)
        print('%11s:[%.2f, %.2f]: %.3f' % (
            function.__name__.capitalize(),
            result['x'], result['y'], result['value']))

    drawPlot(function, limits, results)


runFunction(fnc.sphere, (-5.12, 5.12), HillClimb)
# runFunction(fnc.griewank, (-10, 10), HillClimb)
# runFunction(fnc.rosenbrock, (-10, 10), HillClimb)
# runFunction(fnc.rastrigin, (-5.12, 5.12), HillClimb)
# runFunction(fnc.schwefel, (-500, 500), HillClimb)
# runFunction(fnc.levy, (-10, 10), HillClimb)
# runFunction(fnc.zakharov, (-10, 10), HillClimb)
# runFunction(fnc.michalewicz, (0, np.pi), SimulatedAnnealing)
