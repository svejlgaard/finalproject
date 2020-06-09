import numpy as np
from scipy import stats

from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os, contextlib, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets



# Sets the directory to the current directory
os.chdir(sys.path[0])


#Taken from https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/

plt.rcParams.update({'font.size': 4})

def pearsonr_ci(x, y, alpha=0.05, _print=True):
	''' calculate Pearson correlation along with the confidence interval using scipy and numpy
	Parameters
	----------
	x, y : iterable object such as a list or np.array
	Input for correlation calculation
	alpha : float
	Significance level. 0.05 by default
	Returns
	-------
	r : float
	Pearson's correlation coefficient
	pval : float
	The corresponding p value
	lo, hi : float
	The lower and upper bound of confidence intervals
	'''

	r, p = stats.pearsonr(x,y)
	r_z = np.arctanh(r)
	se = 1/np.sqrt(x.size-3)
	z = stats.norm.ppf(1-alpha/2)
	lo_z, hi_z = r_z-z*se, r_z+z*se
	lo, hi = np.tanh((lo_z, hi_z))
	if _print: print(f"\t corr {r:.3f} in [{lo:.3f}, {hi:.3f}], with p={p:.3f}")
	return lo, hi, p

def corrfunc(x, y, **kws):
	r, _ = stats.pearsonr(x, y)
	ax = plt.gca()
	if not np.isnan(r): ax.annotate("r = {:.2f} € [{:.2f},{:.2f}], p = {:.2f} ".format(r, *pearsonr_ci(x,y,_print=False)), xy=(.1, .9), xycoords=ax.transAxes)

def meanfunc(x, **kws):
	m, s = x.mean(), x.std()
	ax = plt.gca()
	ax.annotate(f"{m:.2f}±{s:.2f}", xy=(.1, .9), xycoords=ax.transAxes)

def summary_analyze(data: pd.DataFrame, filename):
	print('shape', data.shape)
	# Mean and std...
	print(data.describe())

	# Covariance matrix
	for pair in combinations(data.columns, r=2):
		print(f"{pair[0]} and {pair[1]}")
		pearsonr_ci(data[pair[0]], data[pair[1]])
	# print(data.corr())
	# plt.matshow(data.corr())
	# plt.show()

	# Joint and marginal
	plt.figure()
	g = sb.PairGrid(data, palette=list("red"))
	#g.map_diag(sb.distplot, kde=False, bins=10)
	#g.map_diag(meanfunc)
	#g.map_upper(sb.kdeplot, cmap="Blues_d")

	g.map(plt.scatter, s=10)
	g.map(corrfunc)
	sb.set(font_scale=0.4)
	filename = filename.split('_')[2]
	plt.savefig(f'Pearson_{filename}.png')


for i,filename in enumerate(os.listdir('data')):
	dataframe = pd.read_csv(f'data/{filename}')
	summary_analyze(dataframe,filename)
