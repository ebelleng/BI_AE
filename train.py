# Aim : Deep-Learning: Training via BP+GD
# Date: 27-09-21

import pandas     as pd
import numpy      as np
import my_utility as ut
	
# Softmax's training
def train_softmax(x,y,param):
    print(f'x shape: {x.shape}')
    print(f'y shape: {y.shape}')
    nodos_final, N = y.shape
    mu = param[1]
    w, _     = ut.iniW(nodos_final, x)
    costo = []
    for iter in range(param[0]):        
        gW, c = ut.grad_softmax(x,y,w,lambW=param[2])
        
        costo.append(c)
        w = w - mu*gW

    return(w,costo)

# AE's Training 
def train_ae(x,hnode,MaxIter,mu):
    w1,w2 = ut.iniW(hnode,x)

    for iter in range(MaxIter):        
        #Step 1: Forward
        Act = ut.forward_ae(x,w1,w2) 

        #Step2: Backward
        w1, w2 = ut.backward_ae(Act, x, w1, w2, mu) 

    return (w1)
    
#SAE's Training 
def train_sae(x,param):    
    W = []
    data = x
    for hn in range(2,len(param)):
        print('AE={} Hnode={}'.format(hn-1,param[hn]))
        # Se entrena el encoder
        w = train_ae(data, param[hn], param[0], param[1])
        # Se guarda el peso del encoder
        W.append(w)
        # Se calcula la nueva data
        data = ut.act_sigmoid( np.dot(w, data))
        
    return(W,data) 
   
# Beginning ...
def main():
    param_sae,param_sft = ut.load_config()    
    xe              = ut.load_data_csv('train_x.csv')
    print(xe.shape)

    ye              = ut.load_data_csv('train_y.csv')
    W,Xr            = train_sae(xe,param_sae) 
    Ws, cost        = train_softmax(Xr,ye,param_sft)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

