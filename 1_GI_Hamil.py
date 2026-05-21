#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:54:37 2026

@author: Familie_Wenting
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the function we want to approximate

def H(q1, q2, p1, p2, m=1.0, l=1.0, g=9.81):
    delta = q1 - q2
    kinetic = (p1**2 + 2*p2**2 - 2*p1*p2*np.cos(delta)) / (1 + np.sin(delta)**2)
    kinetic *= 1/(2*m*l**2)
    potential = m*g*l*(4 - 2*np.cos(q1) - np.cos(q2))
    return kinetic + potential

# Define the Gaussian Matrix
def gaussian_matrix(X, centers, sigma):
    Phi = np.zeros((len(X), len(centers)))
    for i, c in enumerate(centers):
        diff = X - c
        dist2 = np.sum(diff**2, axis=1)
        Phi[:, i] = np.exp(-dist2/(2*sigma**2))
    return Phi








# Data----------------------------------------------------------------------------------------------------------------------------------------
N = 50

q1 = np.random.uniform(-np.pi, np.pi, N)
q2 = np.random.uniform(-np.pi, np.pi, N)
p1 = np.random.uniform(-5, 5, N)
p2 = np.random.uniform(-5, 5, N)

X = np.column_stack([q1, q2, p1, p2])
Z = H(q1, q2, p1, p2)

# Gaussian centers
num_centers = 250
centers = X[np.random.choice(N, num_centers, replace=True)]
sigma = 2.0


# Calculate the weights
Phi = gaussian_matrix(X, centers, sigma)
w = np.linalg.lstsq(Phi, Z, rcond=None)[0]







# Make the grid for plotting----------------------------------------------------------------------------------------------------------------------------------------
q1_vals = np.linspace(-np.pi, np.pi, 40)
q2_vals = np.linspace(-np.pi, np.pi, 40)

Q1, Q2 = np.meshgrid(q1_vals, q2_vals)

# Flatten grid
q1_flat = Q1.reshape(-1)
q2_flat = Q2.reshape(-1)
p1_flat = np.zeros_like(q1_flat)
p2_flat = np.zeros_like(q1_flat)

X_new = np.column_stack([q1_flat, q2_flat, p1_flat, p2_flat])




# Values for plotting ----------------------------------------------------------------------------------------------------------------------------------------
# Predict the Hamiltonian 
Phi_new = gaussian_matrix(X_new, centers, sigma)
Z_pred = (Phi_new @ w).reshape(Q1.shape)

# True values on grid
Z_true = H(Q1, Q2, 0, 0)

# Calculate the error
error = np.linalg.norm(Z_true - Z_pred) / np.sqrt(Z_pred.size)
print("The normalized error per sample point is:", error)








# Plot----------------------------------------------------------------------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Q1, Q2, Z_true, alpha=0.4, color = 'green')
ax.plot_surface(Q1, Q2, Z_pred, alpha=0.5, color = 'blue')
ax.set_title("True Hamiltonian + Predicted Hamiltonian")

ax.set_xlabel("q1")
ax.set_ylabel("q2")
ax.set_zlabel("H")

plt.show()






