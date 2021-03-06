import numpy as np
import pandas as pd
class softmax_layer:
    def __init__(self,X,n_y,y):
        self.W1=np.random.randn(n_y,X.shape[1])* 0.01
        self.b1=np.zeros((n_y,1))
        self.X=X
        y_h=np.zeros((X.shape[0],n_y))
        for i in range(0,X.shape[0]):
            y_h[i][y[i]]=1
        self.y_h=y_h.T
    def softmax(self,prob):
        return np.exp(prob)/np.sum(np.exp(prob),axis=0,keepdims=True)
        
    def feedforward(self):
        self.Z1 = np.dot(self.W1,self.X.T)+self.b1
        self.A1 = np.exp(self.Z1)/np.sum(np.exp(self.Z1),axis=0,keepdims=True)
        
    def loss(self):
        self.cost=-(1/self.X.shape[0])*np.sum(self.y_h*np.log(self.A1))
        print("cost=",self.cost)
    
    def backflow(self):
        self.dW1=-(1/self.X.shape[0])*np.dot((self.y_h-self.A1),self.X)
        self.db1=-(1/self.X.shape[0])*np.sum(self.y_h - self.A1,axis=1,keepdims=True)
        #print(self.y_h.shape,self.A1.shape,self.X.shape)
        
    def update(self,n):
        self.n=n
        self.W1=self.W1-n*self.dW1
        self.b1=self.b1-n*self.db1
    
    def model(self,no_of_iterations,n):
        for i in range(0,no_of_iterations):
            self.feedforward()
            self.loss()
            self.backflow()
            self.update(n)
        return self.accuracy()
