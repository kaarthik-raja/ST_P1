
import numpy as np
from numpy.linalg import pinv as matinv
from numpy.linalg import det as matdet
from sklearn.cluster import KMeans

LogNConst = np.log(np.pi*2)/(-2)
NConst   = np.divide(1,np.power(np.pi*2,19))
eps = np.finfo(float).eps
y=None
x=None

class KMean:

	def __init__(self,n_clusters=512, random_state=0):
		self.n_clusters=n_clusters
		self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
		print(n_clusters,"n_clustersS")
		
	def meanvar(self,X):
		lbl = self.kmeans.fit_predict(X)
		centres = self.kmeans.cluster_centers_
		Covar = []
		eps = np.finfo(float).eps
		w = []
		wsum = 0
		for i in range(self.n_clusters):

			Idx = np.where(lbl == i)
			elems = X[Idx[0]]
			wi = len(elems)
			if wi < 3:
				elems = np.append(elems,elems+10*eps,axis=0)
			w.append(wi)
			wsum = wsum + wi
			Covar.append(np.cov(np.transpose(elems)))
		# print(np.array(w)/wsum)
		return centres,np.array(Covar), np.array(w)/wsum

def CT():
	input("Continue:  ")

class GMM:
	def __init__(self,CompNum=512, Mean = None, Covar= None,Weight=None):
		self.CompNum = CompNum

		self.Mean = Mean
		self.Covar = Covar
		self.Weight = Weight

	
	def GMMinit(self,X):
		self.FDim= X.shape[-1]
		if self.FDim != 38:
			print("Error: Feature Dimension for code Hardcoded as 38")
			input("Continue after Changes to Code: ")
		clstrs = KMean(self.CompNum,0)
		self.Mean,self.Covar ,self.Weight= clstrs.meanvar(X)
		print(self.Weight)
		del clstrs
		
	def expectation(self,X):
		for i in range(self.CompNum):
			Mn = self.Mean[i]
			Cv = self.Covar[i]
			CvI = matinv(Cv)

			X_M = np.subtract(X,Mn)   # N*38
			XC  = np.dot(X,CvI)
			XCX = np.multiply(XC,X)   #N*38
			XCX = np.sum(XCX,axis=1)  #N

			#_____________________________________

			CDet = matdet(Cv)
			Wt = self.Weight[i]

			term = np.log(CDet) / (-2) + Wt + LogNConst*self.FDim






			
			# X_M_CvI = np.
		X_M= np.subtract(np.expand_dims(X,-2),self.Mean)  # N * C * 38 from N * 1 * 38 - C * 38
		X_M = np.expand_dims(X_M,-2) # N*C*1*38
		X_M_CvI = np.matmul(X_M , self.CovarI)  #    N*C*1*38 . C*38*38  =   N*C*1*38
		exp = np.multiply(X_M_CvI,X_M) # N*C*1*38
		exp = np.divide(np.sum(exp,axis=(2,3)),2) # N*C
		# exp =np.exp(np.divide(exp,2))
		#############
		term = np.divide(np.log(self.CovarDet),-2) + np.log(self.Weight) + LogNConst    #19 is 38 /2 #C
		# print(exp,term,"\n====",exp.shape,term.shape)
		probs = np.add(exp,term)   #N*C # C
		print(probs,"==",np.argmax(probs,axis=1))
		CT()
		xmax = np.max(probs,axis=1 ,keepdims=True)      #N
		postsum = np.log(np.sum(np.exp(probs-xmax),axis=1))+xmax[:,0] #N


		postsum = np.where(np.isfinite(postsum),postsum,xmax[:,0]) #N

		postprob = np.exp(np.subtract(probs,np.expand_dims(postsum,-1))) #N*C

		# print(postprob.shape,"==========",postprob)
		return postprob,postsum

	def maximisation(self,X,probs):#N*38,N*C



		N = np.sum(probs,axis=0) #C
		print(probs.shape,"W Changes",self.Weight)
		self.Weight = np.divide(N,np.sum(N))

		F  =  np.dot(np.transpose(probs),X)  #C*38
		self.Mean = np.divide(F,np.expand_dims(N,-1))

		for i in range(self.CompNum):
			Pi = probs[:,i:i+1]    #N*1
			Pi_X = np.multiply(X,Pi)  #N*38
			Pi_XX = np.dot(np.transpose(Pi_X),X) #38*38 
			Pi_XX_N = np.divide(Pi_XX,N[i])  #38*38
			self.Covar[i] = np.subtract(Pi_XX_N ,np.outer(self.Mean[i],self.Mean[i]))


	def EM_iter(self,X,N=10):
		if self.Mean is None:
			self.GMMinit(X)
			print(self.Mean)
			CT()

		for i in range(N):
			self.probs,Psum=self.expectation(X)
			self.maximisation(X,self.probs)
		pass

def main():
	global y
	a =np.load('a.npy')
	b =np.load('b.npy')
	c =np.load('c.npy')
	print(np.mean(a,axis=0))
	print(np.mean(b,axis=0))
	print(np.mean(c,axis=0))
	input("Mean continue:")
	X =np.append(a,b,axis=0)
	X =np.append(X,c,axis=0)
	y = GMM(3)
	y.EM_iter(X,2)
	print(y.Mean)

if __name__ == '__main__':
	main()
	