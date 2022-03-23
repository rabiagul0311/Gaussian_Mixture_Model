# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:23:21 2020

@author: W10
"""


# In[3]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[4]:


print(iris.data.shape)
n_samples, n_features = iris.data.shape
print(n_samples)
print(n_features)
print(iris.data[0])
print(iris.target.shape)
print(iris.target)
print(iris.target_names)


# In[5]:


data, target = iris.data, iris.target


# In[6]:


import matplotlib.pyplot as plt

f = plt.figure()    
f, axes = plt.subplots(nrows = 1, ncols = 3, sharex=True, sharey = True,figsize=(18, 4))
# The indices of the features that we are plotting
x_index = 0

axes[0].scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
axes[0].set_xlabel(iris.feature_names[x_index])
axes[0].set_ylabel(iris.feature_names[1])

axes[1].scatter(iris.data[:, 0], iris.data[:, 2], c=iris.target)
axes[1].set_xlabel(iris.feature_names[x_index])
axes[1].set_ylabel(iris.feature_names[2])

axes[2].scatter(iris.data[:, 0], iris.data[:, 3], c=iris.target)
axes[2].set_xlabel(iris.feature_names[x_index])
axes[2].set_ylabel(iris.feature_names[3])

plt.show()


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data[:,:2],target,test_size=0.2,random_state=42)


# In[8]:


import numpy as np
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)

X = x_train

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '***', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

# Plot the winner
color_iter = itertools.cycle(['navy', 'darkorange'])
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                           color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(1)
    splot.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model, 2 components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()


# In[12]:


from matplotlib.colors import LogNorm

for i in range(1,7):
    gmm = mixture.GaussianMixture(n_components =i)
    gmm.fit(x_train)
    y_pred=gmm.predict(x_test)
#Creating  meshgrid and score samples by using trained model 
    X, Y = np.meshgrid(np.linspace(4, 8), np.linspace(1.5,5))
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm.score_samples(XX)
    Z = Z.reshape((50,50))
#Plotting contour curves by using the score samples 
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(x_train[:,0], x_train[:,1])
    plt.title(str(i)+' number of Gaussians')
    plt.show()


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(data[:50,:2],target[:50],test_size=0.2,random_state=42)
print("FOR IRIS SETOSA CLASS GMM ")
for i in range(1,7):
    gmm = mixture.GaussianMixture(n_components =i)
    gmm.fit(x_train)
    y_pred=gmm.predict(x_test)
#Creating  meshgrid and score samples by using trained model 
    X, Y = np.meshgrid(np.linspace(4, 8), np.linspace(1.5,5))
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm.score_samples(XX)
    Z = Z.reshape((50,50))
#Plotting contour curves by using the score samples 
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(x_train[:,0], x_train[:,1])

    plt.title("Gaussian Distrubition for class 1 with {} components".format(i))
    plt.xlabel("sepal lenght")
    plt.ylabel("sepal width")
    plt.show()


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(data[50:100,:2],target[50:100],test_size=0.2,random_state=42)
print("FOR IRIS VERSICOLOR CLASS GMM ")
for i in range(1,7):
    gmm = mixture.GaussianMixture(n_components =i)
    gmm.fit(x_train)
    y_pred=gmm.predict(x_test)
#Creating  meshgrid and score samples by using trained model 
    X, Y = np.meshgrid(np.linspace(4, 8), np.linspace(1.5,5))
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm.score_samples(XX)
    Z = Z.reshape((50,50))
#Plotting contour curves by using the score samples 
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(x_train[:,0], x_train[:,1])
    plt.title("Gaussian Distrubition for class 2 with {} components".format(i))
    plt.xlabel("sepal lenght")
    plt.ylabel("sepal width")
    plt.show()


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(data[100:150,:2],target[100:150],test_size=0.2,random_state=42)
print("FOR IRIS VIRGINICA CLASS GMM")
for i in range(1,7):
    gmm = mixture.GaussianMixture(n_components =i)
    gmm.fit(x_train)
    y_pred=gmm.predict(x_test)
#Creating  meshgrid and score samples by using trained model 
    X, Y = np.meshgrid(np.linspace(4, 8), np.linspace(1.5,5))
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm.score_samples(XX)
    Z = Z.reshape((50,50))
#Plotting contour curves by using the score samples 
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(x_train[:,0], x_train[:,1])
    plt.title("Gaussian Distrubition for class 3 with {} components".format(i))
    plt.xlabel("sepal lenght")
    plt.ylabel("sepal width")
    plt.show()

