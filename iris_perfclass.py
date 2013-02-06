from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

classifiers = [RidgeClassifier, SGDClassifier, SVC, KNeighborsClassifier]

data = load_iris()
x = np.concatenate((data['data'],data['target'].reshape((150,1))),axis=1)
np.random.shuffle(x)
train = x[:100,:]
test = x[100:,:]

for classifier in classifiers:
    classify = classifier()
    classify.fit(train[:,:4],train[:,4])
    score = classify.score(test[:,:4],test[:,4])
    print classify.__repr__(), score