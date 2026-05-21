#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 09:22:43 2026

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


# Gaussian gradient
def gaussian_gradient(X, centers, w, sigma):
    N, D = X.shape
    grad = np.zeros((N, D))

    for i, c in enumerate(centers):
        diff = X - c
        dist2 = np.sum(diff**2, axis=1)
        phi = np.exp(-dist2 / (2 * sigma**2))

        grad += w[i] * phi[:, None] * (-diff / sigma**2)

    return grad


# Data----------------------------------------------------------------------------------------------------------------------------------------
N = 1000

q1 = np.random.uniform(-np.pi, np.pi, N)
q2 = np.random.uniform(-np.pi, np.pi, N)
p1 = np.random.uniform(-5, 5, N)
p2 = np.random.uniform(-5, 5, N)

X = np.column_stack([q1, q2, p1, p2])
Z = H(q1, q2, p1, p2)

# Gaussian centers
num_centers = 1000
centers = X[np.random.choice(N, num_centers, replace=False)]
sigma = 2.0

# Calculate the weights
Phi = gaussian_matrix(X, centers, sigma)
w = np.linalg.lstsq(Phi, Z, rcond=None)[0]





# Grid----------------------------------------------------------------------------------------------------------------------------------------
n = 40
q1_vals = np.linspace(-np.pi, np.pi, n)
q2_vals = np.linspace(-np.pi, np.pi, n)

Q1, Q2 = np.meshgrid(q1_vals, q2_vals)

q1_flat = Q1.ravel()
q2_flat = Q2.ravel()
p1_flat = np.zeros_like(q1_flat)
p2_flat = np.zeros_like(q1_flat)

X_new = np.column_stack([q1_flat, q2_flat, p1_flat, p2_flat])



# Values for plotting ----------------------------------------------------------------------------------------------------------------------------------------
# Predict the Hamiltonian
Phi_new = gaussian_matrix(X_new, centers, sigma)
Z_pred = (Phi_new @ w).reshape(n, n)
Z_true = H(Q1, Q2, 0, 0)


# gradient prediction
grad_pred = gaussian_gradient(X_new, centers, w, sigma)

grad_q1 = grad_pred[:, 0].reshape(n, n)
grad_q2 = grad_pred[:, 1].reshape(n, n)


# true gradient (p1=p2=0)
dH_dq1 = 2 * 9.81 * np.sin(q1_flat)
dH_dq2 = 9.81 * np.sin(q2_flat)

dH_dq1 = dH_dq1.reshape(n, n)
dH_dq2 = dH_dq2.reshape(n, n)



# Plot ----------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(121)
plt.quiver(Q1, Q2, dH_dq1, dH_dq2)
plt.title("True Gradient")

plt.subplot(122)
plt.quiver(Q1, Q2, grad_q1, grad_q2)
plt.title("Predicted Gradient")

plt.show()


# Calculate the error----------------------------------------------------------------------------------------------------------------------------------------
error = np.linalg.norm(Z_true - Z_pred) / np.sqrt(Z_pred.size)

diff_q1 = dH_dq1 - grad_q1
diff_q2 = dH_dq2 - grad_q2
total_error_q1 = np.sum(np.abs(diff_q1))
total_error_q2 = np.sum(np.abs(diff_q2))
#error per point is (diff_q1,diff_q2)

#L2 norm
grad_error = np.sqrt(diff_q1**2 + diff_q2**2)


# Plot Error----------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(6,5))
plt.quiver(Q1, Q2, diff_q1, diff_q2)
plt.title("Gradient Error Vector Field")
plt.show()

plt.figure(figsize=(6,5))
plt.contourf(Q1, Q2, grad_error, levels=20)
plt.colorbar()
plt.title("Gradient Error Heatmap")
plt.xlabel("q1")
plt.ylabel("q2")
plt.show()




total_error = np.sum(grad_error)
mean_error = np.mean(grad_error)
max_error = np.max(grad_error)
print("Mean gradient error:", mean_error)
print("Max gradient error:", max_error)