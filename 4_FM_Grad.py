#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 09:29:25 2026

@author: Familie_Wenting
"""

import numpy as np
import matplotlib.pyplot as plt


# Hit-and-Run algorithm
def one_shot_hit_and_run(u, D, Dr, L0, L1, d_trans):

    u_max = np.max(u, axis=1)
    u_min = np.min(u, axis=1)

    # ensure each dimension crosses zero
    if np.any(np.sign(u_min * u_max) > 0):
        raise ValueError("Each dimension must have both negative and positive values")

    w_in = np.zeros((Dr, D))
    b_in = np.zeros((Dr, 1))

    for j in range(Dr + d_trans):

        b = L0 + (L1 - L0) * np.random.rand()

        s = np.sign(np.random.randn(D))

        dr = s * np.abs(np.random.randn(D))
        dr = dr / np.linalg.norm(dr)

        x_m = np.where(s > 0, u_min, u_max)
        x_p = np.where(s > 0, u_max, u_min)

        denom_m = np.dot(dr, x_m)
        denom_p = np.dot(dr, x_p)

        # avoid division by zero
        if abs(denom_m) < 1e-10 or abs(denom_p) < 1e-10:
            continue

        w0m = (L0 - b) / denom_m
        w1p = (L1 - b) / denom_p

        vs = np.sort([w1p, w0m])
        w = vs[0] if vs[0] > 0 else 1e5

        dw = w * np.random.rand()

        if j >= d_trans:
            ss = np.sign(np.random.randn())
            w_in[j - d_trans, :] = ss * dw * dr
            b_in[j - d_trans, 0] = ss * b

    return w_in, b_in


# Hamiltonian function
def H(q1, q2, p1, p2, m=1.0, l=1.0, g=9.81):
    delta = q1 - q2
    kinetic = (p1**2 + 2*p2**2 - 2*p1*p2*np.cos(delta)) / (1 + np.sin(delta)**2)
    kinetic *= 1/(2*m*l**2)
    potential = m*g*l*(4 - 2*np.cos(q1) - np.cos(q2))
    return kinetic + potential


# define Activation function
def activation(z):
    return np.tanh(z)


#Data----------------------------------------------------------------------------------------------------------------------------------------
N = 500

q1 = np.random.uniform(-np.pi, np.pi, N)
q2 = np.random.uniform(-np.pi, np.pi, N)
p1 = np.random.uniform(-5, 5, N)
p2 = np.random.uniform(-5, 5, N)

X = np.column_stack([q1, q2, p1, p2])   # (N, 4)
Z = H(q1, q2, p1, p2)                   # (N,)

# transpose
X_sampler = X.T








# Sample random features----------------------------------------------------------------------------------------------------------------------------------------
D = 4
Dr = 500
L0, L1 = 0.4, 3.8
d_trans = 100

w_in, b_in = one_shot_hit_and_run(X_sampler, D, Dr, L0, L1, d_trans)

# Build feature matrix
Phi = np.zeros((N, Dr))

for j in range(Dr):
    Phi[:, j] = activation(X @ w_in[j] + b_in[j, 0])







# Least squares
alpha, *_ = np.linalg.lstsq(Phi, Z, rcond=None)


# Create grid for visualization (p1 = p2 = 0)
n = 50
q1_vals = np.linspace(-np.pi, np.pi, n)
q2_vals = np.linspace(-np.pi, np.pi, n)

Q1, Q2 = np.meshgrid(q1_vals, q2_vals)

q1_flat = Q1.ravel()
q2_flat = Q2.ravel()
p1_flat = np.zeros_like(q1_flat)
p2_flat = np.zeros_like(q1_flat)

X_new = np.column_stack([q1_flat, q2_flat, p1_flat, p2_flat])







# Values for plotting ---------------------------------------------------------------------------------------------
# true gradient 
dH_dq1 = 2 * 9.81 * np.sin(q1_flat)
dH_dq2 = 9.81 * np.sin(q2_flat)

dH_dq1 = dH_dq1.reshape(n, n)
dH_dq2 = dH_dq2.reshape(n, n)
   

# approximated gradient
grad_q1 = np.zeros(X_new.shape[0])
grad_q2 = np.zeros(X_new.shape[0])

for j in range(Dr):
    w = w_in[j]
    b = b_in[j, 0]

    z = X_new @ w + b
    phi_prime = 1 - np.tanh(z)**2

    grad_q1 += alpha[j] * phi_prime * w[0]
    grad_q2 += alpha[j] * phi_prime * w[1]

# reshape
grad_q1 = grad_q1.reshape(n, n)
grad_q2 = grad_q2.reshape(n, n)





#Plot-------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(12,5))

# True gradient
plt.subplot(121)
plt.quiver(Q1, Q2, dH_dq1, dH_dq2)
plt.title("True Gradient Field")

# Predicted gradient
plt.subplot(122)
plt.quiver(Q1, Q2, grad_q1, grad_q2)
plt.title("Predicted Gradient Field")
plt.show()







# Error-------------------------------------------------------------------------------------------------------------------------------
#compute error per point: = Gradient true Hamiltonian - Gradient predicted hamiltonian
diff_q1 = dH_dq1 - grad_q1
diff_q2 = dH_dq2 - grad_q2
total_error_q1 = np.sum(np.abs(diff_q1))
total_error_q2 = np.sum(np.abs(diff_q2))
#error per point is (diff_q1,diff_q2)

#L2 norm
grad_error = np.sqrt(diff_q1**2 + diff_q2**2)

plt.figure(figsize=(6,5))
plt.quiver(Q1, Q2, diff_q1, diff_q2)
plt.title("Gradient Error Vector Field")
plt.show()






#Plot Error------------------------------------------------------------------------------------------------------
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