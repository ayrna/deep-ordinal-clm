import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

x = np.linspace(-5, 5, 100)
logit = stats.logistic.cdf(x, 0, 1)
probit = stats.norm.cdf(x, 0, 1)
cloglog = 1 - np.exp(-np.exp(x))

plt.style.use('ggplot')
plt.plot()
plt.plot(x, logit, label='Logit', color='#B5D5C5')
plt.plot(x, probit, label='Probit', color='#B08BBB')
plt.plot(x, cloglog, label='Complementary log-log', color='#ECA869')

plt.legend()
plt.savefig("plot.png")
plt.savefig("plot.pdf")