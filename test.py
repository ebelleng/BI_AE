import pandas as pd
import numpy as np
import my_utility as ut

   
# Feed-forward of the DL
def forward_dl(x,W):
	data = x
	for w in W:
		print(f'data dim: {data.shape}')
		print(f'w dim: {w.shape}')
		z = np.dot(w,data)
		data = ut.softmax(z)

	zv = data
    
	return(zv)

# Beginning ...
def main():		
	xv     = ut.load_data_csv('test_x.csv')	
	yv     = ut.load_data_csv('test_y.csv')	
	W      = ut.load_w_dl()
	zv     = forward_dl(xv,W)      		
	ut.metricas(yv,zv) 
	

if __name__ == '__main__':   
	 main()

