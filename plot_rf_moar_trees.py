import numpy as np
from matplotlib import pyplot as plt

tree_counts = [5, 10, 25, 50, 100]

acc = [0.96, 0.9721739130434782, 0.97,  0.9717391304347827, 0.9695652173913043]
prec = [ 0.9135254988913526, 0.9651972157772621, 0.9361233480176211, 0.9484304932735426, 0.9340659340659341]

cv_score = [0.9640869565217391, 0.9668695652173913, 0.9726956521739132, 0.974434782608695, 0.9747826086956521]

plt.figure()
plt.plot(tree_counts, acc)
plt.plot(tree_counts, prec)
plt.plot(tree_counts, cv_score)
plt.legend(['accuracy','precision','CV score'], 'lower right')
plt.xlabel('number of trees')
plt.savefig('vary_tree_acc_prec.png')