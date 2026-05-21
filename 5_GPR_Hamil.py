#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:18:29 2026

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




# Data----------------------------------------------------------------------------------------------------------------------------------------
N = 500

q1 = np.random.uniform(-np.pi, np.pi, N)
q2 = np.random.uniform(-np.pi, np.pi, N)
p1 = np.random.uniform(-5, 5, N)
p2 = np.random.uniform(-5, 5, N)

X = np.column_stack([q1, q2, p1, p2])
Y = H(q1, q2, p1, p2)






# Training covariance Matrix----------------------------------------------------------------------------------------------------------------------------------------
n = X.shape[0]
length = 1

K = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        diff = X[i] - X[j]
        k = length**2 * np.exp(-(np.sum(diff**2)) / (2 * length**2))
        K[i, j] = k
        K[j, i] = k
        




# Add gaussian distributed noise
sigma = 0.1
A = K + sigma**2 * np.eye(n)
alpha = np.linalg.solve(A, Y)# aplha shape is (500,)





# Create grid/test points----------------------------------------------------------------------------------------------------------------------------------------
n = 10
q1_vals = np.linspace(-np.pi, np.pi, n)
q2_vals = np.linspace(-np.pi, np.pi, n)
# p1_vals = np.linspace(-np.pi, np.pi, n)
# p2_vals = np.linspace(-np.pi, np.pi, n)


Q1, Q2 = np.meshgrid(q1_vals, q2_vals)
X_star = np.column_stack([
    Q1.ravel(),
    Q2.ravel(),
    np.zeros(Q1.size),
    np.zeros(Q1.size)
])

n_train = X.shape[0] # here we rename n to avoid confusion same as n before
n_test = Q1.size







# Build test set----------------------------------------------------------------------------------------------------------------------------------------
Q1, Q2 = np.meshgrid(q1_vals, q2_vals)

X_star = np.column_stack([
    Q1.ravel(),
    Q2.ravel(),
    np.zeros(Q1.size),
    np.zeros(Q1.size)])


# Calculate Test covariance matrix   
K_star = np.zeros((n_train, n_test))
for j in range(n_test):
    diff = X - X_star[j]
    K_star[:, j] = length**2 * np.exp(
        -(np.sum(diff**2, axis=1)) / (2 * length**2))
    
    
    
    
    
    
    
    




# Posterior mean + variance ------------------------------------------------------------------------------------------------------------------------------------
posterior_mean = K_star.T @ alpha
posterior_mean = K_star.T @ alpha
B = np.linalg.solve(A, K_star)
posterior_variance = ( length**2 - np.sum(K_star * B, axis=0))
std = np.sqrt(posterior_variance)

posterior_mean = posterior_mean.reshape(Q1.shape)
std = std.reshape(Q1.shape)







# Plot------------------------------------------------------------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Q1, Q2, posterior_mean, alpha=0.4, color = 'magenta')
ax.set_title("Posterior mean + variance")

# highest and lowest variance surfaces
ax.plot_surface(Q1, Q2, posterior_mean + 2*std, alpha=0.3, color = 'blue')
ax.plot_surface(Q1, Q2, posterior_mean - 2*std, alpha=0.3, color = 'green')


mean_variance = np.mean(posterior_variance)
print(mean_variance)





