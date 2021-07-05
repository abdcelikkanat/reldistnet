# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:37:32 2021

@author: nnak
"""



# Import all the packages
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

import torch_sparse
CUDA = torch.cuda.is_available()
from Z_Initialization import Z_Initialization
from sklearn import metrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
undirected=1





class LSM(nn.Module,Z_Initialization):
    def __init__(self,sparse_i,sparse_j, input_size,latent_dim,AR_order=3,MATRIX_TYPE='diagonal',initialization=None,scaling=None):
        super(LSM, self).__init__()
        Z_Initialization.__init__(self)
        self.input_size=input_size
        self.MATRIX_TYPE=MATRIX_TYPE 
        self.AR_order=AR_order
       
        self.latent_dim=latent_dim
        self.initialization=1
        self.scaling=1
        #create indices to index properly the receiver and senders variable
       
        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.CUDA=True
        self.pdist_tol1=nn.PairwiseDistance(p=2,eps=0)
        
      
          
        self.edge_i_idx=sparse_i
        self.edge_j_idx=sparse_j
        
        self.spectral_data=self.spectral_clustering()#.flip(1)

    # PARAMETERS
        # node-specific bias
        self.gamma=nn.Parameter(torch.randn(self.input_size,device=device)).to(device)
        # AR coefficients
        self.AR_bias=nn.Parameter(torch.randn(self.input_size,latent_dim,device=device)).to(device)
        if self.MATRIX_TYPE=='diagonal':
            self.Alpha=nn.Parameter(torch.randn(AR_order,latent_dim))
        elif self.MATRIX_TYPE=='full':
            self.Alpha=nn.Parameter(torch.randn(AR_order,latent_dim,latent_dim))
        else:
            raise ValueError("Wrong Matrix Type Input String.")


      
        # spectral_centroids_to_z=self.spectral_data
        if self.spectral_data.shape[1]>latent_dim:

            self.latent_z=nn.Parameter(self.spectral_data[:,0:latent_dim]).to(device)
        elif self.spectral_data.shape[1]==latent_dim:
            self.latent_z=nn.Parameter(self.spectral_data)
        else:
            self.latent_z=nn.Parameter(torch.zeros(self.input_size,latent_dim,device=device))
            self.latent_z.data[:,0:self.spectral_data.shape[1]]=self.spectral_data
            
            
    def static_init(self):
        '''

        Returns
        -------
        log_likelihood : The Poisson likelihood for t=0 static initialization of the network

        '''
        dist_mat=torch.exp(-((self.latent_z.unsqueeze(1)-self.latent_z+1e-06)**2).sum(-1)**0.5)
        non_link_likelihood=0.5*torch.mm(torch.exp(self.gamma.unsqueeze(0)),(torch.mm((dist_mat-torch.diag(torch.diagonal(dist_mat))),torch.exp(self.gamma).unsqueeze(-1))))
                
        #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
        link_likelihood=(-((((self.latent_z[self.edge_i_idx]-self.latent_z[self.edge_j_idx]+1e-06)**2).sum(-1)))**0.5+self.gamma[self.edge_i_idx]+self.gamma[self.edge_j_idx]).sum()
        
               

        
        log_likelihood=link_likelihood-non_link_likelihood
        
        return log_likelihood
    
    def init_latent_series(self):
        '''
        Initialize Z_0,Z_-1,Z_-2,...
        Based on the static treatment of the first Graph snapshot
        '''
        self.Z_series_init=nn.Parameter(torch.zeros(self.input_size,self.AR_order,self.latent_dim,device=device))
        self.Z_series_init.data=self.latent_z.data.unsqueeze(1).repeat_interleave(self.AR_order,1)
        self.Latent_Z_t=self.Z_series_init.data
    def create_AR_timestep(self):
        '''
        UPDATE lagged matrix so it containes the lagged inputs.
        '''
        
        self.Latent_Z_t=torch.cat((self.Latent_Z_t[:,1:,:],self.Z_t.data.unsqueeze(1)),1)
    
    
    def AR_Process(self,timestep):
        '''
        Zt...T should be given as->NxtxD
        '''
        
        if self.MATRIX_TYPE=='diagonal':
            # Alpha -> 1xtxD
            #Zt...T->NxtxD
            #Z_t->NxD
            if timestep==0:
                self.Z_t=(self.Z_series_init*self.Alpha.unsqueeze(0)).sum(1)+self.AR_bias
         
            else:
                self.Z_t=(self.Latent_Z_t*self.Alpha.unsqueeze(0)).sum(1)+self.AR_bias
            
            
        if self.MATRIX_TYPE=='full':
            # Alpha -> txDXD
            #Zt...T->Nxtx1xD
            #Z_t->NxD
            if timestep==0:
                self.Z_t=(self.Alpha.mm(self.Z_series_init.unsqueeze(2))).sum(1).squeeze(1)+self.AR_bias

           
            else:
                    
                self.Z_t=(self.Alpha.mm(self.Latent_Z_t.unsqueeze(2))).sum(1).squeeze(1)+self.AR_bias
            
            
    def forward(self):
        '''

        Returns
        -------
        log_likelihood : The Poisson likelihood for t=1 static initialization of the network

        '''
        dist_mat=torch.exp(-((self.Z_t.unsqueeze(1)-self.Z_t+1e-06)**2).sum(-1)**0.5)
        non_link_likelihood=0.5*torch.mm(torch.exp(self.gamma.unsqueeze(0)),(torch.mm((dist_mat-torch.diag(torch.diagonal(dist_mat))),torch.exp(self.gamma).unsqueeze(-1))))
                
        #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
        link_likelihood=(-((((self.Z_t[self.edge_i_idx]-self.Z_t[self.edge_j_idx]+1e-06)**2).sum(-1)))**0.5+self.gamma[self.edge_i_idx]+self.gamma[self.edge_j_idx]).sum()
        
               

        
        log_likelihood=link_likelihood-non_link_likelihood
        
        return log_likelihood
    def link_prediction(self,i_idx,j_idx):
        idx=torch.triu_indices(self.input_size, self.input_size,1)
        i=idx[0]
        j=idx[1]
       
        with torch.no_grad():
            dist_mat=torch.exp(-((self.Z_t.unsqueeze(1)-self.Z_t+1e-06)**2).sum(-1)**0.5)
          
            rates=torch.exp(self.gamma)[i]*(dist_mat-torch.diag(torch.diagonal(dist_mat)))[i,j]*torch.exp(self.gamma)[j]
            self.rates=rates
            targets_full=torch.zeros(self.input_size,self.input_size)
            targets_full[i_idx,j_idx]=1
            target=targets_full[i,j]
            
            #fpr, tpr, thresholds = metrics.roc_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())
            precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(tpr,precision)
    

        
with open("C:/Users/nnak/Discrete_time_LSM/networks/diverging_clusters.pkl", 'rb') as f:
    data = pickle.load(f)
     
edges=data['edges']

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


sparse_i=torch.from_numpy(np.array(edges[0])[:,0]).long().to(device)
sparse_j=torch.from_numpy(np.array(edges[0])[:,1]).long().to(device)
# mask=sparse_i<sparse_j
# sparse_i=sparse_i[mask]
# sparse_j=sparse_j[mask]
N=8
latent_dim=2    
   
model = LSM(sparse_i,sparse_j,N,latent_dim=latent_dim).to(device)
#model = LSM(torch.randn(N,latent_dim),sparse_i,sparse_j,N,latent_dim=latent_dim,CVflag=False,graph_type='undirected',missing_data=False).to(device)
        
optimizer = optim.Adam(model.parameters(), 0.01)  

# Optimize static case network for t=0
for epoch in range(1000):
                  
    loss=-model.static_init()
    optimizer.zero_grad() # clear the gradients.   
    loss.backward() # backpropagate
    optimizer.step() 
# Initialize lagged Z_0,Z
model.init_latent_series()

#timestep loop
T=10
for t in range(len(edges)):
    print(t)
    model.edge_i_idx=torch.from_numpy(np.array(edges[t])[:,0]).long().to(device)
    model.edge_j_idx=torch.from_numpy(np.array(edges[t])[:,1]).long().to(device)
    if t>0:
        model.create_AR_timestep()
    for epoch in range(1000):
        model.AR_Process(timestep=t)

        loss=-model.forward()
        optimizer.zero_grad() # clear the gradients.   
        loss.backward() # backpropagate
        optimizer.step() 
        if epoch%100==0:
            print(loss.data)
            ROC,PR=model.link_prediction(model.edge_i_idx,model.edge_j_idx)
            print(ROC)
        
        

    








