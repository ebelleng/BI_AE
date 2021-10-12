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
    n0, _ = x.shape # 256, 1600
    n1 = hn

    r = np.sqrt(6 / ( n1 + n0))
    w1 = np.random.rand(n1, n0) * 2* r - r    #dim -> (400x256)
    w2 = np.random.rand(n0, n1) * 2* r - r    #dim -> (256x400)
       
    return (w1,w2)

# STEP 1: Feed-forward of AE
def forward_ae(x,w1,w2):	
    # Salida del encoder
    z1 = np.dot(w1, x)
    a1 = act_sigmoid(z1)
    # Salida del decoder
    z2 = np.dot(w2, a1)
    a2 = act_sigmoid(z2)
    
    return (a1, a2)

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
def updW_ae(w1,w2,dCdW,mu):
    gW1,gW2 = dCdW
    w1-= mu*gW1
    w2-= mu*gW2
    return(w1,w2)

# Softmax's gradient
def grad_softmax(x,y,w,lambW):
    z = np.exp( np.dot(w, x) )
    a = softmax(z)
    _, N = y.shape

    Cost = (-1/N) * np.sum( np.sum( y * np.log10(a) ) ) 
    Cost += lambW/2 * np.linalg.norm(w, ord=2) ** 2

    gW = ((-1/N) * (np.dot((y-a),np.transpose(x)))) + (lambW * w)

    return(gW,Cost)

# Calculate Softmax
def softmax(z):
    z = np.exp(z)
    return z / z.sum(axis=0)

# Métrica
def metricas(x,y):
    #dados 2 arreglos crea la matriz de confusion y calcula las matericas
    #actua = [1,0,1,0,0,0,0,1,1,1,2,3,4,5,6,7,8,9]
    #predi = [0,1,0,0,0,0,0,1,1,1,2,3,4,5,6,7,8,9]
    #cm = confusion_matrix(actua,predi)
    cm = confusion_matrix(x,y)
    
    #completar cm
    print('Matriz de confusion: \n',cm)
   # return 
    #complete code  
    f_score = []
    for i in range(10):
        pre = precison(cm,i)
        #print('precision: ', pre)
        rec = recall(cm,i)
        #print('recall: ',rec)
        ppr = pre*rec
        #print(ppr)
        pmr = pre+rec
        #print(pmr)
        f = 2*ppr/pmr
        print('F-score',i+1,': ',f)
        f_score = np.append(f_score,f)
        
        
    avgFscore = (1/10)*np.sum(f_score)
    print('avg F-score: ',avgFscore)
 
    save_metricas_dl(avgFscore,f_score)    

def precison(mc,r):
    
    m = mc[r,r]
    #print('m= ',m)
    #print('m= ',mc[r,1])
    sumMC = np.sum(mc[r,:])
    #print(sumMC)
    pre = m/sumMC
    return pre
    
def recall(mc,c):
        
    m = mc[c,c]
    #print('m= ',m)
    #print('m= ',mc[r,1])
    sumMC = np.sum(mc[:,c])
    if(sumMC == 0):
        sumMC = 1/1000
    #print('rec sum:',sumMC)
    rec = m/sumMC
    #print('rec: ',rec)
    return rec 

def save_metricas_dl(avgF, fscore):   
    fscore = np.append(fscore,avgF)
    archivo = open('estima_dl.csv', 'w')

    [archivo.write(f'{fscore[i]},') for i in range(len(fscore)) ]

    
    archivo.close()
    
#Confusuon matrix
def confusion_matrix(x,y):
    #complete code  

    classes = np.unique(x) # extract the different classes
    cm = np.zeros((len(classes), len(classes))) # initialize the confusion matrix with zeros

    for i in range(len(classes)):
        for j in range(len(classes)):

            cm[i, j] = np.sum((x == classes[i]) & (y == classes[j]))
   
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
    keys = [f'w{i}' for i in range(1,len(W)+1) ]
    w = dict(zip(keys, W))     
    # Guardar pesos
    np.savez_compressed('w_dl.npz', **w, ws=Ws ) 
    load_w_dl()

    archivo = open('costo_softmax.csv', 'w')
    [ archivo.write(f'{c}\n') for c in cost ]
    archivo.close()
   
#load weight of the DL 
def load_w_dl():
    W = []
    [ W.append(np.load('w_dl.npz')[w]) for w in np.load('w_dl.npz').files ]
    
    return (W)    
