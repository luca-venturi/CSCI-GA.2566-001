# Python3

import numpy as np
import pickle
import matplotlib.pyplot as plt

# load data - cv 

with open ('data/cv_data', 'rb') as _file:
    [score, rhoRange, tRange, best_rho, best_T] = pickle.load(_file)
kRange = [-int(np.log2(rho)) for rho in rhoRange]

# plot

plt.title('10-Fols Cross-Validation error vs. $\\rho$')
plt.xlabel('$k = -\log_2(\\rho)$')
plt.ylabel('misclassification fraction')
for T in tRange:
	tmp = [score[rho,T] for rho in rhoRange]
	plt.plot(kRange,tmp,label='T = '+str(T))
plt.legend()
plt.savefig('cv', bbox_inches='tight')
plt.show() #

# load data - test

with open ('data/test_data', 'rb') as _file:
    [score,] = pickle.load(_file) 

# plot

plt.title('Test error vs. $T$')
plt.xlabel('$T$')
plt.ylabel('misclassification fraction')

tmp = [score['m',T] for T in tRange]
plt.plot(tRange,tmp,label='mAdaBoost')

tmp = [score['c',T] for T in tRange]
plt.plot(tRange,tmp,label='AdaBoost')

plt.legend()
plt.savefig('test', bbox_inches='tight')
plt.show() #

# load data - cv comp

with open ('data/cv_comp_data', 'rb') as _file:
    [score,] = pickle.load(_file) 

# plot

plt.title('10-Fols Cross-Validation error vs. $T$')
plt.xlabel('$T$')
plt.ylabel('misclassification fraction')

tmp = [score['m',T] for T in tRange]
plt.plot(tRange,tmp,label='mAdaBoost')

tmp = [score['c',T] for T in tRange]
plt.plot(tRange,tmp,label='AdaBoost')

plt.legend()
plt.savefig('cv_comp', bbox_inches='tight')
plt.show() #
