import numpy as np
import math
import matplotlib.pyplot as plt
#from sklearn import svm, datasets
from sklearn import svm, datasets

"""

GOAL: In this exercise we will explore different polynomial kernel degrees for SVM.
      We will use an existing implementation of SVM: the SVC class from sklearn.svm.
      This class solves soft-margin SVM problem.

"""


def plot_results(models, titles, X, y, plot_sv=False):
    """
    Input:
    - models (list): A list of fitted estimators. That is, each element of the list is
                      a return value of the fit method of the SVM model.
    - titles (list): A list of names that corresponds to the models above.
    - X (np.ndarray): Data points in R^2 - A numpy array of shape nx2, where n is the number of data points.
    - y (np.ndarray): A numpy array of labels in {-1,1} for the data points.

    The method plots the data points and the classifiers' prediction.

    """

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100



# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1,X2],axis=1)
y = np.concatenate([np.ones((n,1)), -np.ones((n,1))], axis=0).reshape([-1])





#Code for question 1a
"""
Train 3 soft SVM models with regularization parameter C=10 using:
(1) Linear kernel
(2) homogeneous polynomial kernel of degree 2
(3) homogeneous polynomial kernel of degree 3
"""

clf1 = svm.SVC(C=10, kernel='linear')
clf1.fit(X, y)

clf2 =svm.SVC(C=10, kernel='poly', degree=2, gamma=1, coef0=0.0)
clf2.fit(X,y)


clf3 = svm.SVC(C=10, kernel='poly', degree=3, gamma=1, coef0=0.0)
clf3_fitted=clf3.fit(X,y)

classifiers=[clf1, clf2, clf3]
names=['linear kernel', 'homogenous polynomial kernel of degree 2', 'homogenous polynomial kernel of degree 3']
plot_results(classifiers, names, X, y) #plot results





#Code for question 1b
"""
Train 2 soft SVM models with regularization parameter C=10 using:
(1) non-homogeneous polynomial kernel of degree 2
(2) non-homogeneous polynomial kernel of degree 3
"""

#coef0=1

clf2 =svm.SVC(C=10, kernel='poly', degree=2, gamma=1, coef0=1)
clf2.fit(X,y)


clf3 = svm.SVC(C=10, kernel='poly', degree=3, gamma=1, coef0=1)
clf3_fitted=clf3.fit(X,y)

classifiers=[clf2, clf3]
names=['non-homogenous polynomial kernel of degree 2', 'non-homogenous polynomial kernel of degree 3']
plot_results(classifiers, names, X, y) #plot results





#Code for question 1c

#Perturb the lables: Change each negative label to a positive one with probability 0.1
minus1=np.where(y==-1)
replace=np.random.choice([-1.,1.], len(minus1[0]), p=[0.9,0.1])
y[minus1]=replace

#Train a soft-SVM model a with regularization parameter C=10, using a non-homogeneous polynomial kernel of degree 2
clf2 =svm.SVC(C=10, kernel='poly', degree=2, gamma=1, coef0=1)
clf2.fit(X,y)

#Train a soft-SVM model with a with regularization parameter C=10, using RBF kernel with gamma=10
clf_RBF =svm.SVC(C=10, kernel='rbf',gamma=10)
clf_RBF.fit(X,y)

classifiers=[clf2, clf_RBF]
names=['non-homogenous polynomial kernel of degree 2', 'RBF kernel with gamma=10']
plot_results(classifiers, names, X, y) #plot results


#Explore what happens if we change gamma


#RBF kernel with gamma=0.1
clf_RBF01=svm.SVC(C=10, kernel='rbf',gamma=0.1)
clf_RBF01.fit(X,y)

#RBF kernel with gamma=1
clf_RBF1 =svm.SVC(C=10, kernel='rbf',gamma=1)
clf_RBF1.fit(X,y)

#RBF kernel with gamma=10
clf_RBF10 =svm.SVC(C=10, kernel='rbf',gamma=10)
clf_RBF10.fit(X,y)

#RBF kernel with gamma=50
clf_RBF50 =svm.SVC(C=10, kernel='rbf',gamma=50)
clf_RBF50.fit(X,y)

#RBF kernel with gamma=100
clf_RBF100 =svm.SVC(C=10, kernel='rbf',gamma=100)
clf_RBF100.fit(X,y)


classifiers=[clf_RBF01, clf_RBF1, clf_RBF10, clf_RBF50, clf_RBF100]
names=['gamma=0.1', 'gamma=1','gamma=10', 'gamma=50', 'gamma=100']
plot_results(classifiers, names, X, y) #plot results
