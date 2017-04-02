#just a quick example of GaussianNB using sklearn

import numpy as np
from sklearn.naive_bayes import GaussianNB

#X = feature, Y = label
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

#defining a Gaussian Naive classifier
clf = GaussianNB()
clf.fit(X, Y)
GaussianNB(priors=None)

#asking classifier for some predictions
print(clf.predict([[-0.8, -1]]))
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

GaussianNB(priors=None)
print(clf_pf.predict([[-0.8, -1]]))
