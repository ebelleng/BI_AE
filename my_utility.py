'''
INTEGRANTES
ETIENNE BELLENGER HERRERA   17619315-8
JUAN IGNACIO AVILA OJEDA    19013610-8
'''
# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# Initialize weights
def iniW(hn,x):
    size_input, _ = x.shape # 375, 5

    r = np.sqrt(6 / ( hn + size_input))
    w1 = np.random.randn((hn, size_input)) * 2* r - r    #dim -> (20x5)
    w2 = np.random.randn(size_output, hn) * r           #dim -> (1x20)
       
    return w1,w2

# STEP 1: Feed-forward of AE
def forward_ae(x,w1,w2):	
    # Calcula la activación de los Nodos Ocultos
    z1 = np.dot(w1, x)
    a1 = act_sigmoid(z1)
    # Calcula la activación de los Nodos de Salida
    z2 = np.dot(w2, a1)
    a2 = act_sigmoid(z2)

    return (a1,a2)

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# STEP 2: Feed-Backward
def backward_ae(Act, x, w1, w2, mu):
    # Valor calculado
    _, a2 = Act 
    # Calcular el error
    error = a2 - x
    # Calcular el gradiente oculto y salida    
    dCdW = gradW_ae(Act, x, w2, error)
    # Actualizar los pesos
    w1, w2 = updW_ae(w1, w2, dCdW, mu)
    # Calcular Error cuadratico medio
    mse = np.sum(error[0] ** 2) / len(error[0])
    return (w1,w2)

def gradW_ae(Act,x,w2,e):    
    a1, a2 = Act
    z2 = deriva_sigmoid(a2) # 1, 375
    z1 = deriva_sigmoid(a1) # 20, 375
    
    # Calcular gradiente decoder
    delta2 = np.multiply(e, deriva_sigmoid(z2)) # Probar con a2
    dCdW2 = np.dot(delta2, a1.T)
    # Calcular gradiente 
    delta1 = np.multiply( np.dot(w2.T, delta2), deriva_sigmoid(z1) )
    dCdW1 = np.dot( delta1, x.T)

    return dCdW1, dCdW2

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
    return
   
#load weight of the DL 
def load_w_dl():
    #complete code
    return(W)    
