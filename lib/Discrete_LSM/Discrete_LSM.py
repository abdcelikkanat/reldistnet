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
torch.autograd.set_detect_anomaly(True)






class LSM(nn.Module,Z_Initialization):
    def __init__(self,graph_snaps,W_ijT,input_size,latent_dim,AR_order=3,MATRIX_TYPE='full',initialization=None,scaling=None,reg_strength=1):
        super(LSM, self).__init__()
        Z_Initialization.__init__(self)
        self.input_size=input_size
        self.MATRIX_TYPE=MATRIX_TYPE 
        self.AR_order=AR_order
        self.softmax=nn.Softmax(dim=1)
        self.latent_dim=latent_dim
        self.initialization=1
        self.scaling=1
        self.reg_strength=reg_strength
        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.CUDA=True
        self.pdist_tol1=nn.PairwiseDistance(p=2,eps=0)
        
        self.a=nn.Parameter((torch.randn(10,10)))

          
        self.graph_snaps=graph_snaps
        self.W_ijT=W_ijT
        self.T=len(self.graph_snaps)
        
        self.spectral_data=self.spectral_clustering()#.flip(1)

    # PARAMETERS
        # node-specific bias
        self.gamma=nn.Parameter(torch.randn(self.input_size,device=device)).to(device)
        # AR coefficients
        self.AR_bias=nn.Parameter(torch.randn(self.input_size,latent_dim,device=device)).to(device)
        # Alpha arranged as [-t,...,-1,0]
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
        log_likelihood : The Poisson likelihood for t=0 static (aggregated snapshots) initialization of the network

        '''
        dist_mat=torch.exp(-((self.latent_z.unsqueeze(1)-self.latent_z+1e-06)**2).sum(-1)**0.5)
        non_link_likelihood=0.5*self.T*torch.mm(torch.exp(self.gamma.unsqueeze(0)),(torch.mm((dist_mat-torch.diag(torch.diagonal(dist_mat))),torch.exp(self.gamma).unsqueeze(-1))))
                
        #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
        link_likelihood=(W_ijT._values()*(-((((self.latent_z[self.W_ijT._indices()[0]]-self.latent_z[self.W_ijT._indices()[1]]+1e-06)**2).sum(-1)))**0.5+self.gamma[self.W_ijT._indices()[0]]+self.gamma[self.W_ijT._indices()[1]])).sum()
        
               

        
        log_likelihood=link_likelihood-non_link_likelihood
       
        
        return log_likelihood
    
    def init_latent_series(self):
        '''
        Initialize Z_-t, ..., Z_-1, Z_0
        Based on the static treatment of the first Graph snapshot
        '''
        self.Z_series=nn.Parameter(torch.zeros(self.input_size,self.AR_order,self.latent_dim,device=device))
        self.Z_series.data=self.latent_z.data.unsqueeze(1).repeat_interleave(self.AR_order,1)
   
    
    
    def AR_Process(self):
        '''
        Zt...T should be given as->NxtxD
        '''
        self.latent_zetas=[]
        log_lik=0
        if self.MATRIX_TYPE=='diagonal':
            # required dimensions
            # Alpha -> 1xtxD
            # latent_zetas->NxtxD
            # prediction Z_t->NxD
            
            # loop over the t index denoting the timestamp of the network
            for t in range(self.T-1):
                # indeces denoting edges of the network for timestamp t
                edge_i=self.graph_snaps[t]._indices()[0]
                edge_j=self.graph_snaps[t]._indices()[1]
                if t==0:
                    # initial t=0 uses the initialized Z_series: -1, ..., -ar_order
                    Z_t=(self.Z_series*self.softmax(self.Alpha).unsqueeze(0)).sum(1)+self.AR_bias
                    
                    # append prediction to the list to be used as the time series for t=t+1
                    self.latent_zetas.append(Z_t.unsqueeze(1))
                    
                    # calculate the likelihood for t through the forward pass
                    log_lik=self.forward(Z_t,edge_i,edge_j)
                    
                elif t<self.AR_order:
                    # in this case we use Z_series and exchanging part of the time series with the previous prediction
                    Z_t=(self.Z_series[:,t:,:]*self.softmax(self.Alpha)[0:self.AR_order-t,:].unsqueeze(0)).sum(1)+(torch.cat(self.latent_zetas,1)*self.softmax(self.Alpha)[self.AR_order-t:self.AR_order,:].unsqueeze(0)).sum(-1)+self.AR_bias
                    self.latent_zetas.append(Z_t.unsqueeze(1))
                    log_lik=log_lik+self.forward(Z_t,edge_i,edge_j)-self.reg_strength*(((self.latent_zetas[-2].squeeze(1)-self.latent_zetas[-1].squeeze(1))**2).sum(-1)**0.5).sum()

                    
                else:
                    # Normal Update rule
                    Z_t=(torch.cat(self.latent_zetas[t-self.AR_order:t],1)*self.softmax(self.Alpha).unsqueeze(0)).sum(1)+self.AR_bias
                    self.latent_zetas.append(Z_t.unsqueeze(1))
                    log_lik=log_lik+self.forward(Z_t,edge_i,edge_j)-self.reg_strength*(((self.latent_zetas[-2].squeeze(1)-self.latent_zetas[-1].squeeze(1))**2).sum(-1)**0.5).sum()

            
            
        if self.MATRIX_TYPE=='full':
            # Alpha -> txDXD
            # latent_zetas->->Nxtx1xD
            # prediction Z_t->NxD
           
            for t in range(self.T):
                edge_i=self.graph_snaps[t]._indices()[0]
                edge_j=self.graph_snaps[t]._indices()[1]
                if t==0:
                    # initial t=0 uses the initialized Z_series: -1, ..., -ar_order

                    Z_t=(torch.matmul(self.Z_series.unsqueeze(2),self.Alpha)).sum(1).squeeze(1)+self.AR_bias
                    self.latent_zetas.append(Z_t.unsqueeze(1))
                    log_lik=self.forward(Z_t,edge_i,edge_j)
                elif t<self.AR_order:
                    # in this case we use Z_series and exchanging part of the time series with the previous prediction

                    Z_t=torch.matmul(self.Z_series.unsqueeze(2)[:,t:,:].view(self.input_size,-1,1,self.latent_dim),self.Alpha[0:self.AR_order-t].view(-1,self.latent_dim,self.latent_dim)).sum(1).squeeze(1)
                    Z_t=Z_t+torch.matmul(torch.cat(self.latent_zetas,1).unsqueeze(2).view(self.input_size,-1,1,self.latent_dim),self.Alpha[self.AR_order-t:self.AR_order].view(-1,self.latent_dim,self.latent_dim)).sum(1).squeeze(1)+self.AR_bias
                    self.latent_zetas.append(Z_t.unsqueeze(1))

                    log_lik=log_lik+self.forward(Z_t,edge_i,edge_j)-self.reg_strength*(((self.latent_zetas[-2].squeeze(1)-self.latent_zetas[-1].squeeze(1))**2).sum(-1)**0.5).sum()

                    
                else:
                    Z_t=torch.matmul(torch.cat(self.latent_zetas[t-self.AR_order:t],1).unsqueeze(2).view(self.input_size,-1,1,self.latent_dim),self.Alpha).sum(1).squeeze(1)+self.AR_bias
                    self.latent_zetas.append(Z_t.unsqueeze(1))

                    log_lik=log_lik+self.forward(Z_t,edge_i,edge_j)-self.reg_strength*(((self.latent_zetas[-2].squeeze(1)-self.latent_zetas[-1].squeeze(1))**2).sum(-1)**0.5).sum()                    
                    
                    

        return log_lik

            
            
    def forward(self,latent_z_t,edge_i_idx,edge_j_idx):
        '''

        Returns
        -------
        log_likelihood : The Poisson likelihood for t=1 static initialization of the network

        '''
        dist_mat=torch.exp(-((latent_z_t.unsqueeze(1)-latent_z_t+1e-06)**2).sum(-1)**0.5)
        non_link_likelihood=0.5*torch.mm(torch.exp(self.gamma.unsqueeze(0)),(torch.mm((dist_mat-torch.diag(torch.diagonal(dist_mat))),torch.exp(self.gamma).unsqueeze(-1))))
                
        #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
        link_likelihood=(-((((latent_z_t[edge_i_idx]-latent_z_t[edge_j_idx]+1e-06)**2).sum(-1)))**0.5+self.gamma[edge_i_idx]+self.gamma[edge_j_idx]).sum()
        
               

        
        log_likelihood=link_likelihood-non_link_likelihood
        
        return log_likelihood
    
    def link_prediction(self,i_idx,j_idx,t=None):
        idx=torch.triu_indices(self.input_size, self.input_size,1)
        i=idx[0]
        j=idx[1]
        if t==None:
            with torch.no_grad():
                dist_mat=torch.exp(-((self.latent_z.unsqueeze(1)-self.latent_z+1e-06)**2).sum(-1)**0.5)
              
                rates=torch.exp(self.gamma)[i]*(dist_mat-torch.diag(torch.diagonal(dist_mat)))[i,j]*torch.exp(self.gamma)[j]
                self.rates=rates
                targets_full=torch.zeros(self.input_size,self.input_size)
                targets_full[i_idx,j_idx]=1
                target=targets_full[i,j]
                
                #fpr, tpr, thresholds = metrics.roc_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())
                precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())
        else:
            if self.MATRIX_TYPE=='diagonal':
                with torch.no_grad():
                    Z_t=(torch.cat(self.latent_zetas[t-self.AR_order:t],1)*self.softmax(self.Alpha).unsqueeze(0)).sum(1)+self.AR_bias
                    dist_mat=torch.exp(-((Z_t.unsqueeze(1)-Z_t+1e-06)**2).sum(-1)**0.5)
                  
                    rates=torch.exp(self.gamma)[i]*(dist_mat-torch.diag(torch.diagonal(dist_mat)))[i,j]*torch.exp(self.gamma)[j]
                    self.rates=rates
                    targets_full=torch.zeros(self.input_size,self.input_size)
                    targets_full[i_idx,j_idx]=1
                    target=targets_full[i,j]
                    precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())
            else:
                 with torch.no_grad():
                    Z_t=torch.matmul(torch.cat(self.latent_zetas[t-self.AR_order:t],1).unsqueeze(2).view(self.input_size,-1,1,self.latent_dim),self.Alpha).sum(1).squeeze(1)+self.AR_bias
                    dist_mat=torch.exp(-((Z_t.unsqueeze(1)-Z_t+1e-06)**2).sum(-1)**0.5)
                  
                    rates=torch.exp(self.gamma)[i]*(dist_mat-torch.diag(torch.diagonal(dist_mat)))[i,j]*torch.exp(self.gamma)[j]
                    self.rates=rates
                    targets_full=torch.zeros(self.input_size,self.input_size)
                    targets_full[i_idx,j_idx]=1
                    target=targets_full[i,j]
                    precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())
           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(tpr,precision)
    
#LIST containg the graph snapshots
graph_snaps_numpy=[]   
graph_snaps_torch=[]    
 
# Weighted adjacency over time
W_ijT=0
# with open("C:/Users/nnak/Downloads/synthetic_K50_C5_T100000.pkl/synthetic_K50_C5_T100000.pkl", 'rb') as f:
#     data = pickle.load(f)
     
# edges=data['edges']


# for t in range(len(edges)):
#     edge_t_idx=torch.from_numpy(np.array(edges[t]).transpose()).long().to(device)
#     #graph
#     y_ijt = torch.sparse_coo_tensor(edge_t_idx, torch.ones(edge_t_idx.shape[1]), (N, N))
#     graph_snaps.append(y_ijt)
#     if t==0:
#         W_ijT=y_ijt
#     else:
#         W_ijT+=y_ijt
        


for t in range(10,18):
    graph_snaps_numpy.append(np.loadtxt(f'C:/Users/nnak/Discrete_time_LSM/networks/infection/timestamp_{t}.txt',delimiter=' '))
N=410
latent_dim=2   



for t in range(len(graph_snaps_numpy)):
    edge_t_idx=torch.from_numpy(graph_snaps_numpy[t].transpose()).long().to(device)
    #graph
    y_ijt = torch.sparse_coo_tensor(edge_t_idx, torch.ones(edge_t_idx.shape[1]), (N, N))
    graph_snaps_torch.append(y_ijt)
    if t==0:
        W_ijT=y_ijt
    if t<len(graph_snaps_numpy)-1:
        W_ijT+=y_ijt
        





model = LSM(graph_snaps_torch,W_ijT,N,latent_dim=latent_dim).to(device)
#model = LSM(torch.randn(N,latent_dim),sparse_i,sparse_j,N,latent_dim=latent_dim,CVflag=False,graph_type='undirected',missing_data=False).to(device)
        
optimizer = optim.Adam(model.parameters(), 0.01)  

# Optimize static case network for t=0
for epoch in range(1000):
                  
    loss=-model.static_init()
    optimizer.zero_grad() # clear the gradients.   
    loss.backward() # backpropagate
    optimizer.step() 
    if epoch%100==0:
        print(loss.data)

        print(model.link_prediction(W_ijT._indices()[0], W_ijT._indices()[1]))

print('Initialization Done')
plt.scatter(model.latent_z[:,0].detach().cpu().numpy(),model.latent_z[:,1].detach().cpu().numpy())
model.init_latent_series()

for epoch in range(10000):
    loss=-model.AR_Process()
    optimizer.zero_grad() # clear the gradients.   
    loss.backward() # backpropagate
    optimizer.step() 
    if epoch%100==0:
        print(loss.data)
        idx=7
        print(model.link_prediction(graph_snaps_torch[idx]._indices()[0], graph_snaps_torch[idx]._indices()[1],t=idx))
for t in range(len(graph_snaps_torch)-1):
    plt.scatter(model.latent_zetas[t].squeeze(1)[:,0].detach().cpu().numpy(),model.latent_zetas[t].squeeze(1)[:,1].detach().cpu().numpy())
    plt.show()
    
Z_t=(torch.cat(model.latent_zetas[t-model.AR_order:t],1)*model.softmax(model.Alpha).unsqueeze(0)).sum(1)+model.AR_bias
plt.scatter(Z_t.squeeze(1)[:,0].detach().cpu().numpy(),Z_t.squeeze(1)[:,1].detach().cpu().numpy())
plt.show()
    







