"""
Melvin Mai
CSC292
Dr Robert Jacobs
22 November 2022
Problem 1
"""

import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt 

# Data sets for parts a - g
data_X_a = np.array([45])
data_X_b = np.array([43, 44, 45])
data_X_c = np.array([37, 42, 45])
data_X_d = np.array([15, 35, 45])
data_X_e = np.array([37, 45])
data_X_f = np.array([37, 40, 42, 45])
data_X_g = np.array([37, 38, 40, 40, 41, 42, 43, 45])

# y test interval for a
y_a = np.linspace(38, 52, (52-38+1), endpoint=True)

# y test interval for b - g
y_rest = np.linspace(0, 100, 101, endpoint=True)

# Get the viable hypothesis spaces
def viable_hSpace(maxHypSize, dataMin, dataMax, data):
    hSpace = []
    # Create hypothesis spaces in a range of test interval
    # Based on pseudocode from Dr. Jacobs
    for hMin in range(100 + 1):
        for sizeIndex in range(1, maxHypSize + 1):
            hMax = hMin + sizeIndex - 1
            if hMax > 100:
                break
            if hMin < dataMin or hMax > dataMax:
                continue
            else:
                viable_space = [hMin, hMax]
            
            hSpace.append(viable_space)

    space = []
    # Get the viable hypothesis spaces based on data
    for interv in hSpace:
        # Flagging to make sure all data points fits in a hypothesis interval
        flag = 0
        for x in data:        # Cycle through the data
            if (x >= interv[0]) and (x <= interv[1]):
                flag = flag + 1
        if flag == len(data):
            space.append(interv)
    # Return Viable Hypothesis spaces respect to data set
    return space

# Create the likelihood distribution
def likelihood_function(test_space, data, c):
    # Get viable hypothesis spaces based on data
    v_hSpace = viable_hSpace(c, int(min(test_space)), int(max(test_space)), data)
    likelihood = []

    for y in test_space:
        likeli = 0
        # Check each hypothesis interval
        for interv in v_hSpace:
            card = interv[1] - interv[0] + 1 # Cardinality
            # Check if the test variable, y, is in viable hypothesis space
            if int(y) >= interv[0] and int(y) <= interv[1]:
                likeli = likeli + (1 / (card ** len(data)))

        likelihood.append(likeli)

    # Return normalized likelihood function
    return likelihood/np.max(likelihood)

# Plot function
def plot(test_space, likelihood, data, title):
    fig = plt.figure()
    x_axis = np.linspace(min(test_space), max(test_space), len(test_space), endpoint=True)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(x_axis, likelihood, 'black', label="Mean")

    # Plot the marker for each x in data set
    for x in data:
        axes.plot(x, 0, marker = 'o', markersize = 5, markeredgecolor = "black", markerfacecolor="black")

    axes.set_xlabel('X')
    axes.set_ylabel('p(y is in C|X)')
    plt.title(title)
    plt.show()

def main():
    parta = likelihood_function(y_a, data_X_a, 6)
    partb = likelihood_function(y_rest, data_X_b, 40)
    partc = likelihood_function(y_rest, data_X_c, 40)
    partd = likelihood_function(y_rest, data_X_d, 40)
    parte = likelihood_function(y_rest, data_X_e, 40)
    partf = likelihood_function(y_rest, data_X_f, 40)
    partg = likelihood_function(y_rest, data_X_g, 40)

    plot(y_rest, partg, data_X_g, "Part g")

main()


