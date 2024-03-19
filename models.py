import torch
import torch.nn as nn
import torch_geometric as pyg
from utils import *
from copy import deepcopy

class GCNConv(nn.Module):
	def __init__(self, inc, outc):
		super().__init__()
		self.inc, self.outc = inc, outc
		self.feat_transform = nn.Linear(inc,outc)
		self.feat_transform.reset_parameters()
	def forward(self, X, prop_matrix):
		out = prop_matrix @ self.feat_transform(X) 					
		return out

class Trainable_GCN(nn.Module):
	def __init__(self, inc, hidden, outc, dropout = 0.5, use_sage=False):
		super().__init__()
		self.conv1 = GCNConv(inc,hidden)
		self.conv2 = GCNConv(hidden,outc)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.use_sage = use_sage
	def forward(self,X,prop_matrix):
		out = self.dropout(self.activation(self.conv1(X,prop_matrix)))
		out = self.conv2(out,prop_matrix)
		return out
	def fit(self,features, adj, labels, idx_train, idx_val):
		if self.use_sage:
			print('Training Graph Sage.')
		else:
			print('Training GCN.')
		
		if self.use_sage:
			prop = normalize_adj_tensor_sage(adj)
		else:
			prop = normalize_adj_tensor(adj)
		epochs = 200 				
		lr, wd = 1e-2, 5e-4

		optimizer = torch.optim.Adam(self.parameters(),lr=lr, weight_decay=wd)
		criterion = nn.CrossEntropyLoss().cuda()

		best_acc, best_epoch = 0, None
		self.train()

		for epoch in range(1, epochs+1):
			optimizer.zero_grad()
			out = self.forward(features,prop)
			loss = criterion(out[idx_train],labels[idx_train])
			loss.backward()
			optimizer.step()

			train_acc = (out[idx_train].argmax(dim=1) == labels[idx_train]).sum() / len(idx_train)
			val_acc = (out[idx_val].argmax(dim=1) == labels[idx_val]).sum() / len(idx_val)

			if epoch % 10 == 0:
				print('Loss: {:.2f}, Train_acc: {:.2f}, Val_acc: {:.2f}'.format(loss.item(),train_acc,val_acc))

			if val_acc > best_acc:
				best_acc = val_acc
				best_epoch = epoch
				best_state = deepcopy(self.state_dict())
			elif epoch >= best_epoch + 50:
				break

		self.load_state_dict(best_state) # reload best model
	def predict(self,features,adj):
		self.eval()
		if self.use_sage:
			prop = normalize_adj_tensor_sage(adj)
		else:
			prop = normalize_adj_tensor(adj)
		out = self.forward(features,prop)
		return out