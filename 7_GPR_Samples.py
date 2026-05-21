#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:01:43 2026

@author: Familie_Wenting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import r_

# Define the function we want to approximate

def H(q1, q2, p1, p2, m=1.0, l=1.0, g=9.81):
    delta = q1 - q2
    kinetic = (p1**2 + 2*p2**2 - 2*p1*p2*np.cos(delta)) / (1 + np.sin(delta)**2)
    kinetic *= 1/(2*m*l**2)
    potential = m*g*l*(4 - 2*np.cos(q1) - np.cos(q2))
    return kinetic + potential






# Data------------------------------------------------------------------------------------------------------------------------------------
N = 250

q1 = np.random.uniform(-np.pi, np.pi, N)
q2 = np.random.uniform(-np.pi, np.pi, N)
p1 = np.random.uniform(-5, 5, N)
p2 = np.random.uniform(-5, 5, N)

X = np.column_stack([q1, q2, p1, p2])
Y = H(q1, q2, p1, p2)







# Training covariance Matrix------------------------------------------------------------------------------------------------------------------------------------
n = X.shape[0]

K = np.zeros((n, n))
for i in range(n):
    for j in range(i, n): # We only calculate the top diagonal and then later use symmetry to fill in the rest
        diff = X[i] - X[j]
        k = 2 * np.exp(-0.5 * np.sum(diff**2)) # We use the RBF kernel
        K[i, j] = k
        K[j, i] = k


sigma = 0.1
A = K + sigma**2 * np.eye(n)
alpha = np.linalg.solve(A, Y)

# Calulate alpha
alpha = np.linalg.solve(K + sigma**2 * np.eye(n), Y) # aplha shape is (500,)








# Create grid/test points------------------------------------------------------------------------------------------------------------------------------------
n = 10
q1_vals = np.linspace(-np.pi, np.pi, n)
q2_vals = np.linspace(-np.pi, np.pi, n)
p1_vals = np.linspace(-np.pi, np.pi, n)
p2_vals = np.linspace(-np.pi, np.pi, n)

grid = np.meshgrid(q1_vals, q2_vals, p1_vals, p2_vals)
X_star = np.column_stack([g.ravel() for g in grid])


n_train = X.shape[0]
n_test = len(q1_vals)

# training matrix 
A = K + sigma**2 * np.eye(n_train)

# alpha
alpha = np.linalg.solve(A, Y)




# Built test set------------------------------------------------------------------------------------------------------------------------------------
q1_vals = np.linspace(-np.pi, np.pi, 10)

X_star = np.column_stack([
    q1_vals,
    np.zeros_like(q1_vals),
    np.zeros_like(q1_vals),
    np.zeros_like(q1_vals)])

# Test covariance matrix
K_star = np.zeros((n_train, n_test))

for j in range(n_test):
    diff = X - X_star[j]
    K_star[:, j] = 2 * np.exp(-0.5 * np.sum(diff**2, axis=1))








# Posterior mean + variance-----------------------------------------------------------------------------------------------------------------------------------
posterior_mean = K_star.T @ alpha
B = np.linalg.solve(A, K_star)
posterior_variance = 2 - np.sum(K_star * B, axis=0)
std = np.sqrt(posterior_variance)

cov = K_star.T @ np.linalg.solve(A, K_star)
posterior_cov = 2 * np.eye(len(q1_vals)) - cov

#sample functoins
samples = np.random.multivariate_normal(posterior_mean, posterior_cov, 3)









#Plot-----------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,5))

for i in range(3):
    plt.plot(
        q1_vals,
        samples[i],
        color="darkblue",
        alpha=1,
        label="samples" if i == 0 else None
    )

plt.plot(q1_vals, posterior_mean, label="posterior mean", color="magenta") # expected function
plt.fill_between(
    q1_vals,
    posterior_mean - 2*std,
    posterior_mean + 2*std,
    alpha=0.2
    
    )
plt.title("Posterior mean + samples from posterior distribution")

plt.legend()
plt.show()

normalized_variance = posterior_variance / np.max(posterior_variance)

single_value = np.mean(normalized_variance)

print(single_value)