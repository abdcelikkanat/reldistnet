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
import torch_sparse
CUDA = torch.cuda.is_available()
from spectral_clustering import Spectral_clustering_init
from sklearn import metrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
undirected=1





class LSM(nn.Module,Z_Initialization):
    def __init__(self,data,sparse_i,sparse_j, input_size,latent_dim,graph_type,non_sparse_i=None,non_sparse_j=None,sparse_i_rem=None,sparse_j_rem=None,CVflag=False,initialization=None,scaling=None,missing_data=False):
        super(LSM, self).__init__()
        Spectral_clustering_init.__init__(self)
        self.input_size=input_size
        self.cluster_evolution=[]
        self.mask_evolution=[]
        self.init_layer_split=torch.round(torch.log(torch.tensor(data.shape[0]).float()))
        self.init_layer_idx=torch.triu_indices(int(self.init_layer_split),int(self.init_layer_split),1)
       
        self.bias=nn.Parameter(torch.randn(1,device=device)).to(device)
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device)).to(device)
        self.latent_dim=latent_dim
        self.initialization=1
        self.gamma=nn.Parameter(torch.randn(self.input_size,device=device)).to(device)
        self.build_hierarchy=False
        self.graph_type=graph_type
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.flag1=0
        self.sparse_j_idx=sparse_j
        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.missing_data=missing_data
        self.CUDA=True
        self.pdist_tol1=nn.PairwiseDistance(p=2,eps=0)

        
        
      
        self.non_sparse_i_idx_removed=non_sparse_i
     
        self.non_sparse_j_idx_removed=non_sparse_j
           
        self.sparse_i_idx_removed=sparse_i_rem
        self.sparse_j_idx_removed=sparse_j_rem
        self.removed_i=torch.cat((self.non_sparse_i_idx_removed,self.sparse_i_idx_removed))
        self.removed_j=torch.cat((self.non_sparse_j_idx_removed,self.sparse_j_idx_removed))

        
        self.spectral_data=self.spectral_clustering()#.flip(1)

        self.first_centers_sp=torch.randn(int(self.init_layer_split),self.spectral_data.shape[1],device=device)

        global_cl,spectral_leaf_centers=self.kmeans_tree_z_initialization(depth=80,initial_cntrs=self.first_centers_sp) 
           
        self.first_centers=torch.randn(int(torch.round(torch.log(torch.tensor(data.shape[0]).float()))),latent_dim,device=device)
      

        spectral_centroids_to_z=spectral_leaf_centers[global_cl]
        # spectral_centroids_to_z=self.spectral_data
        if self.spectral_data.shape[1]>latent_dim:

            self.latent_z=nn.Parameter(spectral_centroids_to_z[:,0:latent_dim]).to(device)
        elif self.spectral_data.shape[1]==latent_dim:
            self.latent_z=nn.Parameter(spectral_centroids_to_z)
        else:
            self.latent_z=nn.Parameter(torch.zeros(self.input_size,latent_dim,device=device))
            self.latent_z.data[:,0:self.spectral_data.shape[1]]=spectral_centroids_to_z
   
   

 