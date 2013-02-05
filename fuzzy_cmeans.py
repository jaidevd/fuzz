import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from pylab import plot, show, title, figure

def euclidean_dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

class FuzzyCMeans(object):
    
    def __init__(self, data=None, k=3, p=2):
        if data is not None:
            self.data = data
        self.k = k
        self.p = p
        weights = np.random.random((k,self.data.shape[0]))
        self.weights = weights/weights.sum(0)
        
    def fill_distmat(self):
        distmat = np.zeros((self.k, self.data.shape[0]))
        for i in range(distmat.shape[0]):
            for j in range(distmat.shape[1]):
                distmat[i,j] = euclidean_dist(self.centroids[i,:],self.data[j,:])
        self.distmat = distmat
    
    def stepfcm(self):
        mf = self.weights**self.p
        self.centroids = (np.dot(mf,self.data).T/mf.sum(1)).T
        self.fill_distmat()
        tmp = self.distmat**(-2/(self.p-1))
        self.weights = tmp/tmp.sum(0)
    
    def obj_fcn(self):
        return np.sum((self.distmat**2)*(self.weights**self.p))
    
    def run(self,tol=0.01,n_iter=100,show_plot=True):
        i = 0
        self.t = []
        while True:
            self.stepfcm()
            i += 1
            self.t.append(self.obj_fcn())
            if (self.t[len(self.t)-1] < tol) or (i==n_iter):
                break
        if show_plot:
            plot(self.t)
            title('Objective Function')
            show()

if __name__ == "__main__":
    x = load_iris()['data']
    fcm = FuzzyCMeans(data=x)
    fcm.run()
    pca = PCA(n_components=2)
    x_red = pca.fit_transform(x)
    c_red = pca.fit_transform(fcm.centroids)
    figure()
    plot(x_red[:,0],x_red[:,1],'bo')
    plot(c_red[:,0],c_red[:,1],'ro')
    show()