{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 215,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter no of layers\n",
      "4\n",
      "[[1. 1. 0. 0.]]\n",
      "The final cost = 0.009159420330024584\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x7fa9a809fbe0>"
      ]
     },
     "execution_count": 215,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#single layer neural network\n",
    "#solved xor problem using single layer neural network\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "#initialsie neural network\n",
    "#nh no of hidden nodes dim\n",
    "#ny output layer dims\n",
    "#X=(m,nx)\n",
    "#Y=(1,m)\n",
    "#nx x dimesnions\n",
    "#m no of traing examples(d)\n",
    "class NeuralNetwork():\n",
    "    def __init__(self,nx,ny,m,nh):\n",
    "        self.W1=np.random.randn(nh,nx)\n",
    "        self.b1=np.zeros(shape=(nh, 1))\n",
    "        self.W2=np.random.randn(ny,nh)\n",
    "        self.b2=np.zeros(shape=(ny, 1))\n",
    "    #activation\n",
    "    def sigmoid(self,x):\n",
    "        return 1/(1+np.exp(-x))\n",
    "\n",
    "#farword propagation\n",
    "def forward_prop(W1,b1,W2,b2,node):\n",
    "    cache={}\n",
    "    Z1=np.dot(W1,X.T)+b1\n",
    "    A1=node.sigmoid(Z1)\n",
    "    Z2=np.dot(W2,A1)+b2\n",
    "    A2=node.sigmoid(Z2)\n",
    "    cache={\"Z1\":Z1,\n",
    "           \"A1\":A1,\n",
    "           \"Z2\":Z2,\n",
    "           \"A2\":A2}\n",
    "    return cache\n",
    "\n",
    "#back propagation\n",
    "def back_pass(y,cache,node):\n",
    "    m=y.shape[1]\n",
    "    A2=cache[\"A2\"]\n",
    "    A1=cache[\"A1\"]\n",
    "    W2=node.W2\n",
    "    dZ2=A2-y\n",
    "    dW2=np.dot(dZ2,A1.T)/m\n",
    "    db2=1/m*np.sum(dZ2,axis=1,keepdims=True)\n",
    "    dZ1=np.dot(W2.T,dZ2)*A1*(1-A1)\n",
    "    dW1 = 1/m*np.dot(dZ1,X)\n",
    "    db1 = 1/m*np.sum(dZ1,axis=1,keepdims=True)\n",
    "    grads = {\"dW1\": dW1,\n",
    "             \"db1\": db1,\n",
    "             \"dW2\": dW2,\n",
    "             \"db2\": db2}\n",
    "    \n",
    "    return grads\n",
    "    \n",
    "#cost of a function\n",
    "def cost(Y,cache):\n",
    "    A2=cache[\"A2\"]\n",
    "    m=y.shape[1]\n",
    "    loss=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)\n",
    "    return -np.sum(loss)/m\n",
    "\n",
    "#update weights\n",
    "def update(node,grads,n):\n",
    "    node.W1=node.W1-n*grads[\"dW1\"]\n",
    "    node.b1=node.b1-n*grads[\"db1\"]\n",
    "    node.W2=node.W2-n*grads[\"dW2\"]\n",
    "    node.b2=node.b2-n*grads[\"db2\"]\n",
    "\n",
    "#update model\n",
    "def nn_model(node,y,no_of_iterations,learning_rate):\n",
    "    \n",
    "    for i in range(no_of_iterations):\n",
    "        cache=forward_prop(node.W1,node.b1,node.W2,node.b2,node)\n",
    "        loss=cost(y,cache)\n",
    "        grads=back_pass(y,cache,node)\n",
    "        update(node,grads,learning_rate)\n",
    "        \n",
    "    return cache\n",
    "\n",
    "#predict\n",
    "def predictions(node):\n",
    "    cache=forward_prop(node.W1,node.b1,node.W2,node.b2,node)\n",
    "    A2=cache[\"A2\"]\n",
    "    print(np.round(A2))\n",
    "    \n",
    "#data set\n",
    "X=np.array([[1,-1],[-1,1],[1,1],[-1,-1]])\n",
    "y=np.array([[1,1,0,0]])\n",
    "\n",
    "#gettnf dims\n",
    "nx=X.shape[1]\n",
    "m=X.shape[0]\n",
    "print(\"Enter no of layers\")\n",
    "nh=int(input())\n",
    "ny=1\n",
    "\n",
    "node=NeuralNetwork(nx,ny,m,nh)\n",
    "cache=forward_prop(node.W1,node.b1,node.W2,node.b2,node)\n",
    "grads=back_pass(y,cache,node)\n",
    "cache=nn_model(node,y,10000,0.1)\n",
    "predictions(node)\n",
    "print(\"The final cost =\",cost(y,cache))\n",
    "plt.scatter(X[0:,:1],X[0:,1:])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1]\n",
      " [-1]\n",
      " [ 1]\n",
      " [-1]]\n"
     ]
    }
   ],
   "source": [
    "    \n",
    "X=np.array([[1,-1],[-1,1],[1,1],[-1,-1]])\n",
    "print(X[0:,:1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
