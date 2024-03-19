from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import torch_geometric as pyg
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import scipy

def print_write(txt,file):
    print(txt)
    with open(file,'a') as f:
        print(txt,file=f)

def compute_prop(inp_A):
	N = inp_A.shape[0]
	inp_A[np.arange(N),np.arange(N)] = 1 			# addings self loop
	D = np.sum(inp_A,axis=0)
	D = np.abs(D)
	D_invsqrt = 1/np.sqrt(D)
	L = np.eye(inp_A.shape[0]) - D_invsqrt[:,None]*inp_A*D_invsqrt[None,:]
	prop = np.eye(inp_A.shape[0]) - L
	return prop

# same as compute_prop but using differentiable torch operations
def normalize_adj_tensor(adj):
	"""Normalize adjacency tensor matrix.
	"""
	device = adj.device
	mx = adj + torch.eye(adj.shape[0]).to(device)
	rowsum = mx.sum(1)
	r_inv = rowsum.pow(-1/2).flatten()
	r_inv[torch.isinf(r_inv)] = 0.
	r_mat_inv = torch.diag(r_inv)
	mx = r_mat_inv @ mx
	mx = mx @ r_mat_inv
	return mx

def normalize_adj_tensor_sage(adj):
	"""Normalize adjacency tensor matrix.
	"""
	device = adj.device
	mx = adj + torch.eye(adj.shape[0]).to(device)
	rowsum = mx.sum(1)
	r_inv = rowsum.pow(-1).flatten()
	r_inv[torch.isinf(r_inv)] = 0.
	r_mat_inv = torch.diag(r_inv)
	mx = r_mat_inv @ mx
	return mx

def classification_margin(output, true_label):
	probs = torch.softmax(output,dim=0)
	probs_true_label = probs[true_label].clone()
	probs[true_label] = 0
	probs_best_second_class = probs[probs.argmax()]
	return (probs_true_label - probs_best_second_class).item()