# Luca

import numpy as np
from kernel import make_base_kernels

# my_int = Algorithm 3
# Input: 	- features: vector of sample features
#			- labels: vector of sample labels
#			- degree: degree of polynomial combination
#			- lam: KRR parameter
#			- eta: interpolation parameter
#			- beta: optimization problem parameter
#			- eps: tolerance stopping parameter
#			- mu_init: inital value of mu in the iteration
#			- sumbsampling: sumbsampling factor
# Output: 	- mu_prime: found value of mu
#			- poly_ker: final kernel

def get_base_kernels(features, subsampling=1):
    return make_base_kernels(features, subsampling=subsampling)

def find_kernel(x, y, degree=1, lam=10., eta=0.2, L=1., mu0=None, mu_init=None, eps=1e-3, subsampling=1):
    (m, p) = x.shape
    L *= 2.
    m = m / subsampling + int(subsampling > 1)
    mu = np.zeros(p)
    base_kernels = get_base_kernels(x, subsampling=subsampling)
    y = y[::subsampling]
    mu_prime = mu_init
    it = 0
    it_max = 100
    dist = np.linalg.norm(mu - mu_prime)
    while dist > eps and it < it_max:
        mu = mu_prime
        gram = sum_weight_kernels(base_kernels, mu) ** degree + lam * np.eye(m)
        al = np.linalg.solve(gram,y)
        mu_prime = mu - eta * derivatives(degree, base_kernels, mu, al, L)
        mu_prime = mu_prime * (mu_prime > 0.)
        it += 1
        dist_old = dist
        dist = np.linalg.norm(mu - mu_prime)
        if dist > dist_old:
            eta *= 0.8
    print 'L = ', L, 'lam = ', lam 
    print 'iter = ', it
    mu = mu_prime
    base_kernels = get_base_kernels(x, subsampling=1)
    return mu, sum_weight_kernels(base_kernels, mu) ** degree
    
def derivatives(degree, base_kernels, mu, al, L):
    d = []
    tmp = sum_weight_kernels(base_kernels, mu)
    for k in range(mu.size):
        center = (tmp ** (degree -1)) * base_kernels[k, :, :]
        d.append(L * mu[k]  - degree * ((al.T).dot(center)).dot(al))
    return np.array(d)

def sum_weight_kernels(base_kernels, mu):
    tmp = base_kernels.copy()
    for k in range(mu.size):
        tmp[k, :, :] = mu[k] * tmp[k, :, :]
    return np.sum(tmp, 0)
