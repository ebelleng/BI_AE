# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# Initialize weights
def iniW(...):
    #complete code
    return(...)

# STEP 1: Feed-forward of AE
def forward_ae(x,w1,w2):	
    #complete code
	return(...)
#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# STEP 2: Feed-Backward
def gradW_ae(a,x,w1,w2):    
    #complete code
    return(gW1,gW2)    

# Update W of the AE
def updW_ae(w1,w2,gW1,gW2,mu):
    w1-= mu*gW1
    w2-= mu*gW2
    return(w1,w2)

# Softmax's gradient
def grad_softmax(x,y,w,lambW):    
    #complete code    
    return(gW,Cost)

# Calculate Softmax
def softmax(z):
    #complete code          
    return(...)

# Métrica
def metricas(x,y):
    cm     = confusion_matrix(x,y)
    #complete code              
    return(...)
    
#Confusuon matrix
def confusion_matrix(x,y):
    #complete code              
    return(cm)

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the SNN
def load_config():      
    par = np.genfromtxt('cnf_sae.csv',delimiter=',')    
    par_sae=[]
    par_sae.append(np.int16(par[0])) # MaxIter
    par_sae.append(np.float(par[1])) # Learn rate    
    for i in range(2,len(par)):
        par_sae.append(np.int16(par[i]))
    par    = np.genfromtxt('cnf_softmax.csv',delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning 
    par_sft.append(np.float(par[2]))   #Lambda
    return(par_sae,par_sft)
# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    return(x)

# save costo of Softmax and weights SAE 
def save_w_dl(W,Ws,cost):    
    #complete code
   
#load weight of the DL 
def load_w_dl():
    #complete code    
    return(W)    
