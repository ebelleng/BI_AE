# Aim : Deep-Learning: Training via BP+GD
# Date: 27-09-21

import pandas     as pd
import numpy      as np
import my_utility as ut
	
# Softmax's training
def train_softmax(x,y,param):
    w     = ut.iniW(x)
    costo = []
    for iter in range(param[0]):        
        # complete code
        break  
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
    w1,w2 = ut.iniW(len(param), x)
    for hn in range(2,len(param)):
        print('AE={} Hnode={}'.format(hn-1,param[hn]))
        w1 = train_ae(x, len(param), param[0], param[1])
        
        W = []
    return(W,x) 
   
# Beginning ...
def main():
    param_sae,param_sft = ut.load_config()    
    xe              = ut.load_data_csv('train_x.csv')
    ye              = ut.load_data_csv('train_y.csv')
    W,Xr            = train_sae(xe,param_sae) 
    Ws, cost        = train_softmax(Xr,ye,param_sft)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

