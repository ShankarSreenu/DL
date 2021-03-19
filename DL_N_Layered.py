import numpy as np
import pandas as pd
import os
import cv2


#loads the dataset if you input:directory
def load_dataset(folder):
    images = []
    y=[]
    limit=0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            temp=filename.split(".")
            y.append(int(temp[0][-1]))
    return np.array(images),np.array(y)

class neural_network_class:
    def __init__(self,layers,X,Y):
        self.X=X #input 
        self.Y=Y  #classes
        self.m=X.shape[1] #shape of X
        self.y_n=len(np.unique(Y)) #no of classes
        self.no_of_layers=len(layers) # no of layers
        self.parameters={}  #stores W,b,Z,A
        self.d={}   #stores partial derivatives
        self.parameters["A0"]=X #X input can be considered as A0
        layers.append(self.y_n) #appending last layer 
        
        #initialising the parameters weights bias,activation for l layers
        

        for l in range(0,len(layers)-1):
            self.parameters["W"+str(l+1)]=np.random.randn(layers[l+1],layers[l])*0.01
            self.parameters["b"+str(l+1)]=np.zeros((layers[l+1],1))
            self.parameters["Z"+str(l+1)]=None
            self.parameters["A"+str(l+1)]=None
        
        #initialising dZ,dW,db,dA values for l layers
        for l in range(0,len(layers)-1):
            self.d["Z"+str(l+1)]=None
            self.d["W"+str(l+1)]=None
            self.d["b"+str(l+1)]=None
            
    #soft max_function
    #exp(Z1)/sum(exp(Z1))
    def softmax(self):
        A=self.parameters["Z"+str(self.no_of_layers)]
        return np.exp(A)/np.sum(np.exp(A),axis=0,keepdims=True)
    
    
    #one hat encoding
    #example [0,1,2]
    #ouput 
    #[1,0,0]
    #[0,1,0]
    #[0,0,1]
    
    def one_hat_encode(self,Y):
        y_h=np.zeros((Y.shape[0],self.y_n))
        for i in range(0,X.shape[1]):
            y_h[i][Y[i]]=1
        self.y_h=y_h.T

            
    def feedforward(self):
        for l in range(0,self.no_of_layers-1):
            W,b=self.parameters["W"+str(l+1)],self.parameters["b"+str(l+1)]
            A_prev=self.parameters["A"+str(l)]
            Z=np.dot(W,A_prev)+b
            self.parameters["Z"+str(l+1)]=Z
            self.parameters["A"+str(l+1)]=self.Activation(Z,"sigmoid")
        
        #soft max layer
        W,b=self.parameters["W"+str(self.no_of_layers)],self.parameters["b"+str(self.no_of_layers)]
        Aprev=self.parameters["A"+str(self.no_of_layers-1)]
        self.parameters["Z"+str(self.no_of_layers)]=np.dot(W,Aprev)+b
        self.parameters["A"+str(self.no_of_layers)]=self.softmax()
        
    #activation function       
    def Activation(self,Z,arg):
        if arg=="sigmoid":
            return 1/(1+np.exp(-Z))
        if arg=="relu":
            return np.maximum(0,Z)
        
    #calucualate the loss
    #sigma sigma -plogq multi class cross entropy 
    def cost(self):
        A=self.softmax()
        self.loss=(-1/self.m)*np.sum(np.multiply(self.y_h,np.log(A)))
        print(self.loss)
    
    # dervative of sigmoid g(x)(1-g(x))
    def gx(self,x):
        return np.multiply(x,1-x)
        
    #back propagation step
    def backprop(self):
        l=self.no_of_layers
        P = self.parameters["A"+str(l)]
        A_prev=self.parameters["A"+str(l-1)]
        
        #softmax layer
        #partial derivate cost fun
        #predicted P probabities from softmax
        #y_h ground truth
        dZ = (1/self.m)*(P- self.y_h)
        dW=np.dot(dZ,A_prev.T)
        db=np.sum(dZ,axis=1,keepdims=True)
        
        self.d["Z"+str(l)]=dZ
        self.d["W"+str(l)]=dW
        self.d["b"+str(l)]=db
        
        for i in range(0,l-1):
            
            A_prev =   self.parameters["A"+str(l-i-1)]
            W      =  self.parameters["W"+str(l-i)]
            dZ_prev=  self.d["Z"+str(l-i)]
            
            
            dZ     =  np.multiply(np.dot(W.T,dZ_prev),self.gx(A_prev))
            A      =  self.parameters["A"+str(l-i-2)]
            dW     =  np.dot(dZ,A.T)
            db     =  np.sum(dZ,axis=1,keepdims=True)
            #print(dZ)
            self.d["Z"+str(l-i-1)]=dZ
            self.d["W"+str(l-i-1)]=dW
            self.d["b"+str(l-i-1)]=db
            
    #update weights
    #W=W-n*dW
    #b=b-n*db
    def upgrade(self):
        for l in range(0,self.no_of_layers):
            self.parameters["W"+str(l+1)]=self.parameters["W"+str(l+1)]-0.1*self.d["W"+str(l+1)]
            self.parameters["b"+str(l+1)]=self.parameters["b"+str(l+1)]-0.1*self.d["b"+str(l+1)]
        
    #train the model
    def train(self,epochs):
        for i in range(0,epochs):
            network.feedforward()
            network.backprop()
            network.upgrade()
        network.cost()
        
    #caluculates the accuracy 
    def Accuaracy(self):
        i=0
        acc=0
        prob=self.parameters["A"+str(self.no_of_layers)].T
        for val in self.y_h:
            if np.argmax(val)==np.argmax(prob[i]):
                acc=acc+1
            i=i+1
        print((acc/i)*100)
        
    def test(self):
        prediction=self.parameters["A"+str(self.no_of_layers)].T
        y_sample=self.y_h.T
        flattened_labels = np.argmax(y_sample, axis=1) 
        flattened_prediction = np.argmax(prediction, axis=1)
        assert flattened_labels.shape == flattened_prediction.shape
        return np.mean(flattened_labels == flattened_prediction)*100
    
    def test_dataset(self,X,Y):
        self.parameters["A0"]=X
        self.one_hat_encode(Y)
        self.feedforward()
        print(self.test())
    

    
#load train dataset
X,y=load_dataset("/home/shanky/btp/datset/final_train")
Y=np.array(y)
X_flatten = X.reshape(X.shape[0], -1).T
X=X_flatten/255     

        
#you can add no of layers and no of nodes in each layer
layers=[X.shape[0],250,150]
network=neural_network_class(layers,X,Y)
network.one_hat_encode(Y)
network.train(1000)
print(network.test())

#load test dataset
X,y=load_dataset("/home/shanky/btp/datset/final_test")
Y=np.array(y)
X_flatten = X.reshape(X.shape[0], -1).T
X=X_flatten/255
network.test_dataset(X,Y)


