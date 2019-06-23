import random
import matplotlib.pyplot as plt
import csv
import numpy as np

numberOfGenerations = 5000
populationSize = 50
tournamentSize = 2
# mRs = [0.01, 0.02, 0.05, 0.1]
mutationRate = 0.05
numRange = 1  # -range to +range
# Ss = [1e-2, 1e-1, 1, 10]
sigma = 1e-3

def createSamples():
    x = np.arange(0, 10, 0.1)
    noise = np.random.normal(0, 0.01, len(x))
    Y = 8 / 5 + 7 / 25 * x ** 1 - 11 / 50 * x ** 2 + 1 / 50 * x ** 3
    Y += noise
    plt.scatter(x, Y)
    plt.show()
    with open('samples.csv', mode='w') as outFile:
        writer = csv.writer(outFile)
        for i in range(len(Y)):
            writer.writerow([Y[i]])


def fitnessFunction(individual):
    MSE = 0

    for i in range(x.size):
        for j in range(4):
            MSE += (individual[j] * x[i] ** j - Y[i]) ** 2

    MSE /= len(x)
    return 1 / (1 + MSE)


def giveFitnessValue(individual):
    return individual[4]


def mutation(individual):
    for i in range(4):
        p = random.random()
        noise = np.random.normal(0, sigma, 4)
        if p < mutationRate:
            individual[i] += noise[i]


def minMaxAve(population):
    minValue = population[0][4]
    maxValue = minValue
    aveValue = 0

    for i in range(populationSize):
        aveValue += population[i][4]
        minValue = min(minValue, population[i][4])
        maxValue = max(maxValue, population[i][4])

    return [minValue, maxValue, aveValue / populationSize]


# createSamples()
# for sigma in Ss:

# load samples
Y = []
with open('samples.csv', 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        Y.append(float(line[0]))

fig = plt.figure()
x = np.arange(0, 10, 0.1)
plt.scatter(x, Y, color='b')
fig.savefig('samples.png', dpi=fig.dpi)

if True:
    # first population
    population = []
    minMaxAveValues = []
    for i in range(populationSize):
        individual = []
        for j in range(4):
            individual.append((random.random() - 0.5) * 2 * numRange)
        individual.append(fitnessFunction(individual))  # fitnessValue
        population.append(individual)
    minMaxAveValues.append(minMaxAve(population))
    # print(population)
    for nOfG in range(numberOfGenerations):
        if nOfG % 100 == 0:
            print("generation {} of {}".format(nOfG, numberOfGenerations))
            print("maxFitnessValue: {}".format(minMaxAveValues[nOfG][1]))
            print()
        # tournament selection
        parentIndex = []
        for i in range(populationSize):
            sampleIndex = random.sample(range(populationSize), tournamentSize)
            maxIx = 0
            for ix in sampleIndex:
                if population[ix][4] > population[maxIx][4]:
                    maxIx = ix
            parentIndex.append(maxIx)

        # for ix in parentIndex:
        #     print(population[ix])

        # create children
        children = []
        for i in range(populationSize):
            [p1Ix, p2Ix] = random.sample(parentIndex, 2)
            fromP1 = random.sample(range(4), 2)
            child = [0 for o in range(4)]
            for j in range(4):
                if j in fromP1:
                    child[j] = population[p1Ix][j]
                else:
                    child[j] = population[p2Ix][j]
            child.append(0)  # fitnessValue
            children.append(child)
            # print(population[p1Ix])
            # print(population[p2Ix])
            # print(child)
            # print()

        # mutation

        for i in range(len(children)):
            # print(children[i])
            mutation(children[i])
            children[i][4] = fitnessFunction(children[i])
            # print(children[i])
            # print()

        # return
        population.extend(children)

        population.sort(key=giveFitnessValue, reverse=True)
        population = population[:populationSize]
        # for p in population:
        #     print(p)
        minMaxAveValues.append(minMaxAve(population))

    fig = plt.figure()
    plt.scatter(x, Y, color='b')
    xHat = np.arange(0, 10, 0.01)
    ans = population[0]
    YHat = ans[0] + ans[1] * xHat ** 1 + ans[2] * xHat ** 2 + ans[3] * xHat ** 3
    plt.plot(xHat, YHat, color='r')
    plt.show()
    fig.savefig('finalAnswer-mR{}-sigma{}.png'.format(mutationRate, sigma), dpi=fig.dpi)

    print(ans)
    xAx = range(len(minMaxAveValues))
    minMaxAveValues = np.array(minMaxAveValues)
    # print(minMaxAveValues)
    fig = plt.figure()
    plt.scatter(xAx, minMaxAveValues[:, 0])
    fig.suptitle('min-mR{}-sigma{}.png'.format(mutationRate, sigma), fontsize=10)
    plt.show()
    fig.savefig('min-mR{}-sigma{}.png'.format(mutationRate, sigma), dpi=fig.dpi)

    fig = plt.figure()
    plt.scatter(xAx, minMaxAveValues[:, 1])
    fig.suptitle('max-mR{}-sigma{}.png'.format(mutationRate, sigma), fontsize=10)
    plt.show()
    fig.savefig('max-mR{}-sigma{}.png'.format(mutationRate, sigma), dpi=fig.dpi)

    fig = plt.figure()
    plt.scatter(xAx, minMaxAveValues[:, 2])
    fig.suptitle('ave-mR{}-sigma{}.png'.format(mutationRate, sigma), fontsize=10)
    plt.show()
    fig.savefig('ave-mR{}-sigma{}.png'.format(mutationRate, sigma), dpi=fig.dpi)
