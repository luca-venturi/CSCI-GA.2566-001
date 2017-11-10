import numpy as np
import matplotlib.pyplot as plt

# data collected from ls_svm_results

nSV = np.array([740,680,657,630])
nSVm = nSV - np.array([688,620,584,559])

# plot support vectors

d_range = range(1,5)
plt.plot(d_range,nSV,'r',label='num SV')
plt.plot(d_range,nSVm,'b',label='num marginal SV')

plt.title('Number of support vectors vs. degree')
plt.xlabel('degree (d)')
plt.ylabel('count')
plt.legend(loc='center left')
plt.savefig('sk_sv', bbox_inches='tight')
#plt.show()

