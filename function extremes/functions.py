import numpy as np


def sphere(coords):
    sum = 0
    for i in coords:
        sum += i**2

    return sum


def ackley(coords):
    a, b, c = 20.0, 0.2, 2 * np.pi
    sum1, sum2 = 0.0, 0.0
    for i in coords:
        sum1 += i**2
        sum2 += np.cos(c * i)

    d = float(len(coords))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)


def griewank(coords):
    sum, product = 0.0, 0.0
    for i in range(1, len(coords) + 1):
        x = coords[i - 1]
        sum += x**2 / 4000
        product *= np.cos(x**2 / np.sqrt(i))

    return sum - product + 1


def rosenbrock(coords):
    sum = 0.0
    for i in range(1, len(coords)):
        x = coords[i - 1]
        sum += 100 * (coords[i] - x**2)**2 + (x - 1)**2

    return sum


def rastrigin(coords):
    sum = 0.0
    for i in coords:
        sum += i**2 - 10 * np.cos(2 * np.pi * i)
    return 10 * len(coords) + sum


def schwefel(coords):
    sum = 0.0
    for i in coords:
        sum += i * np.sin(np.sqrt(abs(i)))
    return 418.9829 * len(coords) - sum


def levy(coords):
    sum = 0.0
    for i in range(1, len(coords)):
        sum2 = 0.0
        for j in coords:
            sum2 += j

        wi = 1 + (sum2 - 1) / 4
        sum += (wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi)**2)

    xd = coords[len(coords) - 1]
    return np.sin(np.pi * coords[0])**2 + sum + (xd - 1)**2 * (1 + np.sin(2 * np.pi * xd))


def zakharov(coords):
    sum1, sum2 = 0.0, 0.0
    for i in range(1, len(coords) + 1):
        x = coords[i - 1]
        sum1 += x**2
        sum2 += 0.5 * i * x

    return sum1 + sum2**2 + sum2**4


def michalewicz(coords):
    sum = 0.0
    for i in range(1, len(coords) + 1):
        x = coords[i - 1]
        sum += np.sin(x) * np.sin(i * x**2 / np.pi)**20

    return -sum
