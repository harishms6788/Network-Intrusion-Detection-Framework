import numpy as np
from torch import randperm
import time

# Osprey Optimization Algorithm (OOA)
def OOA(X, fitness, lowerbound, upperbound, Max_iterations):
    SearchAgents, dimension = X.shape

    fit = np.zeros((SearchAgents, 1))
    for i in range(SearchAgents):
        fit[i] = fitness(X[i, :])
    best_so_far = np.zeros((Max_iterations, 1))
    average = np.zeros((Max_iterations, 1))
    ct = time.time()

    # main loop
    for t in np.arange(1, Max_iterations + 1).reshape(-1):
        #  update: BEST proposed solution
        Fbest, blocation = np.amin(fit), np.argwhere(fit)
        if t == 1:
            xbest = X[blocation, :]
            fbest = Fbest
        else:
            if Fbest < fbest:
                fbest = Fbest
                xbest = X[blocation, :]
        for i in np.arange(1, SearchAgents + 1).reshape(-1):
            # Phase 1: : POSITION IDENTIFICATION AND HUNTING THE FISH (EXPLORATION)
            fish_position = np.where(fit < fit[i])
            if fish_position.shape[2 - 1] == 0:
                selected_fish = xbest
            else:
                if np.random.rand() < 0.5:
                    selected_fish = xbest
                else:
                    k = randperm(fish_position.shape[1 - 0], 0)
                    selected_fish = X(fish_position[k])
            I = np.round(1 + np.random.rand())
            X_new_P1 = X[i, :] + np.multiply(np.random.rand(1, 1), (selected_fish - np.multiply(I, X[i, :])))
            X_new_P1 = np.amax(X_new_P1, lowerbound)
            X_new_P1 = np.amin(X_new_P1, upperbound)
            # update position based on Eq (6)
            L = X_new_P1
            fit_new_P1 = fitness(L)
            if fit_new_P1 < fit[i]:
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1
            # END Phase 1
            # PHASE 2: CARRYING THE FISH TO THE SUITABLE POSITION (EXPLOITATION)
            X_new_P1 = X[i, :] + (lowerbound + np.random.rand() * (upperbound - lowerbound)) / t
            X_new_P1 = np.amax(X_new_P1, lowerbound)
            X_new_P1 = np.amin(X_new_P1, upperbound)
            # update position based on Eq (8)
            L = X_new_P1
            fit_new_P1 = fitness(L)
            if fit_new_P1 < fit[i]:
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1
            # END Phase 2
        best_so_far[t] = fbest
        average[t] = np.mean(fit)

    Best_score = fbest
    Best_pos = xbest
    OOA_curve = best_so_far
    ct = time.time() - ct

    return Best_score, OOA_curve, Best_pos, ct
