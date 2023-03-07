from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
x = stats.tukeylambda.rvs(-0.06, scale=0.5, size=10000,random_state=2018)
fig = plt.figure(figsize=(30, 10))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
gs =stats.norm.rvs(-0.06, size=1000, random_state=2018)
gs_plot = stats.probplot(gs, plot=ax1, rvalue=True)
tl_plot = stats.probplot(x, plot=ax3, rvalue=True)
res = stats.ppcc_plot(x, -1, 1, plot=ax2)
lam_best = stats.ppcc_max(x)
ax2.vlines(lam_best, 0, 1, colors='r', label=f'Expected shape value\nbest_λ={lam_best:.3f}, true is {-0.06}')
ax2.legend(loc='lower right')
print(lam_best)
plt.show()