import torch
import numpy as np
from sklearn.decomposition import PCA
def SVD(P,k):
    U,sigma,VT = torch.linalg.svd(P)
    sigma1 = torch.diag(sigma)
    U = U[:,:k]
    sigma_SVD = sigma1[:k,:k]
    VT = VT[:k,:]
    svd1 = torch.mm(U,sigma_SVD)
    svd = torch.mm(svd1,VT)
    return svd

def PCA_svd(X, k, center=True):
  n = X.size()[0]
  ones = torch.ones(n).view([n,1])
  h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
  H = torch.eye(n) - h
  H = H.cuda()
  X_center =  torch.mm(H.double(), X.double())
  u, s, v = torch.svd(X_center)
  components  = v[:k].t()
  #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
  return components


P = torch.rand(27,32768)
#P_bar = torch.mean(P,1,True)
#print(P_bar)
#P_pca = PCA_svd(P,1)
#print(P_pca)

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

X = P.numpy()
print(X)
X_bar = np.mean(X,1)
print(X_bar)
pca = PCA(n_components=1)
pca.fit(X)
#print(pca.explained_variance_ratio_)
print(pca.transform(X))
