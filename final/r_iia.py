import numpy as np
from kernel import make_base_kernels

def get_base_kernels(features, subsampling=1):
    return make_base_kernels(features, subsampling=subsampling)

def find_kernel(x, y, degree=1, lam=10., eta=0.2, L=1., mu0=None, mu_init=None, eps=1e-3, subsampling=1):
    beta = 0.5 / L
    (m, p) = x.shape
    m = m / subsampling + int(subsampling > 1)
    mu = mu_init
    base_kernels = get_base_kernels(x, subsampling=subsampling)
    gram = sum_weight_kernels(base_kernels, mu) ** degree + lam * np.eye(m)
    y = y[::subsampling]
    al_prime = np.linalg.solve(gram, y)
    al = np.zeros(m)
    it = 0
    it_max = 100
    while np.linalg.norm(al - al_prime) > eps and it < it_max:
        al = al_prime
        mu = beta * derivatives(degree, base_kernels, mu, al)
        gram = sum_weight_kernels(base_kernels, mu) ** degree + lam * np.eye(m)
        al_prime = eta * al + (1. - eta) * np.linalg.solve(gram,y)
        it += 1
    print 'L = ', L, 'lam = ', lam 
    print 'iter = ', it
    base_kernels = get_base_kernels(x, subsampling=1)
    return mu, sum_weight_kernels(base_kernels, mu) ** degree
    
def derivatives(degree, base_kernels, mu, al):
    d = []
    tmp = sum_weight_kernels(base_kernels, mu)
    for k in range(mu.size):
        center = (tmp ** (degree -1)) * base_kernels[k, :, :]
        d.append(degree * ((al.T).dot(center)).dot(al))
    return np.array(d)

def sum_weight_kernels(base_kernels, mu):
    tmp = base_kernels.copy()
    for k in range(mu.size):
        tmp[k, :, :] = mu[k] * tmp[k, :, :]
    return np.sum(tmp, 0)
