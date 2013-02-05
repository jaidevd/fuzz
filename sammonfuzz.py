# Do not refer to any of this! Read about Sammon-fuzzy clustering somewhere!

import numpy as np
from matplotlib.mlab import find
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from pylab import title, plot, show, figure, clf, xlabel, ylabel

class Results(object):
    def __init__(self, v=None, distout=None, f0=None, itr=None, cost=None):
        if v is not None:
            self.v = v
        if distout is not None:
            self.distout = distout
        if f0 is not None:
            self.f0 = f0
        if itr is not None:
            self.itr = itr
        if cost is not None:
            self.cost = cost

class Params(object):
    def __init__(self):
        pass


def alg1(x):
    return (x - np.amin(x))/(np.amax(x)-np.amin(x))

def alg2(x, c):
    
    n,m = x.shape

    var23 = np.random.randint(low=0,high=x.shape[0],size=(c,))
    v = x[var23,:] + 1e-10
    var25 = x[var23+1,:] - 1e-10
    J = []
    itr = 0
    f0 = np.zeros((x.shape[0],c))

    while np.prod(np.max(abs(v-var25),0)):
        itr += 1
        var25 = v
        dist = np.zeros((x.shape[0],c))
        
        for i in range(c):
            dist[:,i] = ((x-v[i,:])**2).sum(1)
        
        m = np.min(dist,axis=1)
        label = np.argmin(dist,axis=1)
        distout = dist**0.5
        
        for i in range(c):
            var23 = find(label==i)
            if len(var23)>0:
                v[i,:] = x[var23,:].mean(0)
            else:
                ind = round(np.randn()*m-1)
                v[i,:] = x[ind,:]
            f0[var23,i] = 1
        
        J.append(np.sum(f0*dist))
    
    f0 = np.zeros((x.shape[0],c))

    for i in range(c):
        var23 = find(label==i)
        f0[var23,i] = 1

    result = Results(v=v,distout=distout,f0=f0,itr=itr,cost=J)
    return result

def alg3(result, m=2):
    
    N = result.f0.shape[0]
    c, n = result.v.shape
    
    fm = result.f0**m
    PC = 1./(N*np.sum(fm))
    fm = (result.f0)*np.log(result.f0)
    CE = -1./N*np.sum(fm)
    
    result.PC = PC
    result.CE = CE
    
    return result




if __name__ == "__main__":
    
    iris = load_iris()
    x = iris['data']
    c = 3
    C = iris['target']
    
    x = alg1(x)
    result = alg2(x,c)
    result = alg3(result)
    
    d1 = np.amax(result.f0,axis=1)
    d2 = np.argmax(result.f0,axis=1)
    Cc = []
        
    for i in range(c):
        Ci = C[find(d2==i)]
        dum1 = np.histogram(Ci,range(c+1))[0]
        dd2 = np.argmax(dum1)
        Cc.append(dd2)

    pca = PCA(n_components=2)
    x_red = pca.fit_transform(x)
    c_red = pca.fit_transform(result.v)
    
    figure()
    clf()
    
    rang = ['r.','g.','b.']
    
    misclass = []
    
    for i in range(c):
        index = find(C==i)
        Cc = np.array(Cc)
        err = find(Cc[d2[index]]!=i)
        eindex = find(err)
        misclass.append(np.sum(err))
        plot(x_red[index,0],x_red[index,1],rang[i])
        plot(x_red[index[eindex],0],x_red[index[eindex],1],'o')
    
    xlabel("PC1")
    ylabel("PC2")
    title("PCA projection")
    show()