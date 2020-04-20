

import torch
import torch.nn as nn
import torch.nn.functional as F

class EGCN(nn.Module):
	def __init__(self,f_in,f_gcn,f_atten,channels=4):
		super(EGCN,self).__init__()
		self.f_in = f_in
		self.f_gcn = f_gcn
		self.f_atten = f_atten
		self.channels = channels
		
		self.w_atten = nn.Parameter(torch.Tensor(self.channels,self.f_atten)) # w_atten
		self.bn = nn.BatchNorm1d(self.channels) # bn
		self.w = nn.Linear(self.f_in,self.f_gcn) # fc
		self.bn2 = nn.BatchNorm1d(self.f_gcn)
		
		# initialize custom parameters
		nn.init.xavier_uniform_(self.w_atten, gain=1.414)
		
	def get_adj(self,bs,E_):
		A = []
		for i in range(bs):
			A.append(torch.matmul(torch.matmul(torch.matmul(E_[i],
					self.w_atten),self.w_atten.t()),E_[i].t()))
		A = torch.sum(torch.stack(A,0),0)
		A_ = F.softmax(A,1).unsqueeze(2).repeat(1,1,self.channels)
		return A_
		
	def get_h(self,A_adj,x):
		H = []
		for k in range(self.channels):
			H.append(self.w(torch.matmul(A_adj[:,:,k],x)))
			
		H = torch.sum(torch.stack(H,0),0)
		H = self.bn2(H) # bn2
		H = torch.tanh(H)
		return H
		
	def forward(self,x,E): 
		
		bs = x.shape[0]
		E_ = E.permute(0,2,1) # [bs,4,bs]
		E_ = self.bn(E_).permute(0,2,1) # [bs,bs,4]
		
		A_ = self.get_adj(bs,E_) # [bs,bs,4]
		A_adj = E_*A_ # [bs,bs,4]
		H = self.get_h(A_adj,x) # [bs,f_gcn]
		
		return H
		
class EGCNBLOCK(nn.Module):
	def __init__(self, f_gcn,f_atten):
		super(EGCNBLOCK, self).__init__()
		self.n_layer = len(f_gcn) - 1 # layers size
		self.layer_stack = nn.ModuleList() # module list

		for i in range(self.n_layer):
			self.layer_stack.append( 
				EGCN(f_gcn[i], f_gcn[i+1], f_atten[i])
			)

	def forward(self, x, E):
		
		for _, egcn_layer in enumerate(self.layer_stack):
			x = egcn_layer(x,E)
			
		return x
		
class MLPBLOCK(nn.Module):
	def __init__(self, n_units):
		super(MLPBLOCK, self).__init__()
		self.n_layer = len(n_units) - 1 # layers size
		self.layer_stack = nn.ModuleList() # module list

		for i in range(self.n_layer):
			self.layer_stack.append(
				nn.Linear(n_units[i], n_units[i+1])
			)

	def forward(self, x):
		
		for _, mlp_layer in enumerate(self.layer_stack):
			x = mlp_layer(x)
			
		return x

class ScaleNet(nn.Module):
	def __init__(self,f_gcn,f_atten,n_units,le_hs,ld_hs,state_dim):
		super(ScaleNet,self).__init__()
		self.f_gcn = f_gcn # [5,16,32,64]
		self.f_atten = f_atten # [16,16,16]
		self.n_units = n_units # [256,128,64,32,8,2]
		self.le_hs = le_hs # lstm_encoder hidden_size
		self.ld_hs = ld_hs # lstm_decoder hidden_size
		self.state_dim = state_dim # state dim
		self.thresold = 1e-6 # eps
		
		self.egcn_block = EGCNBLOCK(f_gcn,f_atten)
		self.lstm_encoder = nn.LSTMCell(self.f_gcn[-1]+self.state_dim, self.le_hs)
		self.lstm_decoder = nn.LSTMCell(self.le_hs,self.ld_hs)
		
		self.mlps = MLPBLOCK(self.n_units)
		
	# initialize (h,c) of lstm
	def init_hidden_lstm(self, bs, hs):
		if torch.cuda.is_available():
			return (torch.randn(bs, hs).cuda(),torch.randn(bs, hs).cuda())
		else :
			return (torch.randn(bs, hs),torch.randn(bs, hs))
		
	def get_edge(self,x):
		bs = x.shape[0]
		channels = x.shape[1]-1
		x_ = x[:,:-1] # [bs,4]
		x_m = x_.unsqueeze(1).repeat(1,bs,1) # [bs,bs,4]
		x_l = x_.unsqueeze(0).repeat(bs,1,1) # [bs,bs,4]
		E = torch.abs(x_m-x_l)
		
		E_D = torch.sum(E,1) # [bs,4]
		for i in range(bs):
			for k in range(channels):
				E[i][i][k] = E_D[i][k] if E_D[i][k]>self.thresold else 1
		return E
		
	def forward(self,x,state):
		# x: [bs,5] bs the number of persons
		# 5: [x,y,vx,vy,a]
		bs = x.shape[0]
		E = self.get_edge(x)
		x = self.egcn_block(x,E)
		x = torch.cat((x,state),1) # [bs,64+state_dim]
		
		le_h,le_c = self.init_hidden_lstm(bs, self.le_hs)
		ld_h,ld_c = self.init_hidden_lstm(bs, self.ld_hs)
		le_h,_ = self.lstm_encoder(x,(le_h,le_c))
		x,_ = self.lstm_decoder(le_h,(ld_h,ld_c))
		x = F.relu(x)
		
		x = F.relu(self.mlps(x))
		
		return x
		
"""
f_gcn = [5,16,32,64]
f_atten = [16,16,16]
n_units = [256,128,64,32,8,2]
le_hs = 256
ld_hs = 256
state_dim =64
		
model = ScaleNet(f_gcn,f_atten,n_units,le_hs,ld_hs,state_dim)
x = torch.randn(17,5)
state = torch.randn(17,64)

ou = model(x,state)
print(ou.shape)
"""
