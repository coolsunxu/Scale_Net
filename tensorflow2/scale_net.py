

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,initializers,activations


class EGCN(keras.Model):
	def __init__(self,f_gcn,f_atten,channels=4):
		super(EGCN,self).__init__()
		self.f_gcn = f_gcn
		self.f_atten = f_atten
		self.channels = channels
		
		# initialize custom parameters
		self.initializer = initializers.GlorotUniform()
		
		self.w_atten = tf.Variable(self.initializer(shape=
			[self.channels,self.f_atten], dtype=tf.float32)) # w_atten
		self.bn = layers.BatchNormalization() # bn
		self.w = layers.Dense(self.f_gcn) # fc
		self.bn2 = layers.BatchNormalization()
		
	def get_adj(self,bs,E_):
		A = []
		for i in range(bs):
			A.append(tf.matmul(tf.matmul(tf.matmul(E_[i],self.w_atten),
					tf.transpose(self.w_atten,[1,0])),tf.transpose(E_[i],[1,0])))
		A = tf.reduce_sum(tf.stack(A,0),0)
		A_ = tf.tile(tf.expand_dims(activations.softmax(A,1),2),[1,1,self.channels])
		return A_
		
	def get_h(self,A_adj,x):
		H = []
		for k in range(self.channels):
			H.append(self.w(A_adj[:,:,k]@x))
			
		H = tf.reduce_sum(tf.stack(H,0),0)
		H = self.bn2(H) # bn2
		H = tf.tanh(H)
		return H
		
	def call(self,x,E): 
		
		bs = x.shape[0]
		E_ = self.bn(E) # [bs,bs,4]
		
		A_ = self.get_adj(bs,E_) # [bs,bs,4]
		A_adj = E_*A_ # [bs,bs,4]
		H = self.get_h(A_adj,x) # [bs,f_gcn]
		
		return H
		
class EGCNBLOCK(keras.Model):
	def __init__(self, f_gcn,f_atten):
		super(EGCNBLOCK, self).__init__()
		self.n_layer = len(f_gcn) - 1 # layers size
		self.layer_stack = [] # module list

		for i in range(self.n_layer):
			self.layer_stack.append( 
				EGCN(f_gcn[i+1], f_atten[i])
			)

	def call(self, x, E):
		
		for _, egcn_layer in enumerate(self.layer_stack):
			x = egcn_layer(x,E)
			
		return x
		
class MLPBLOCK(keras.Model):
	def __init__(self, n_units):
		super(MLPBLOCK, self).__init__()
		self.n_layer = len(n_units) - 1 # layers size
		self.layer_stack = [] # module list

		for i in range(self.n_layer):
			self.layer_stack.append(
				layers.Dense(n_units[i+1])
			)

	def call(self, x):
		
		for _, mlp_layer in enumerate(self.layer_stack):
			x = mlp_layer(x)
			
		return x

class ScaleNet(keras.Model):
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
		self.lstm_encoder = layers.LSTMCell(self.le_hs)
		self.lstm_decoder = layers.LSTMCell(self.ld_hs)
		
		self.mlps = MLPBLOCK(self.n_units)
		
	# initialize (h,c) of lstm
	def init_hidden_lstm(self, bs, hs):
		return (tf.random.normal([bs, hs]),tf.random.normal([bs, hs]))
		
	def get_edge(self,x): # because tensorflow does not support assignment
		bs = x.shape[0]
		channels = x.shape[1]-1
		x_ = x[:,:-1] # [bs,4]
		x_m = tf.tile(tf.expand_dims(x_,1),[1,bs,1]) # [bs,bs,4]
		x_l = tf.tile(tf.expand_dims(x_,0),[bs,1,1]) # [bs,bs,4]
		E = tf.abs(x_m-x_l).numpy()
		
		E_D = E.sum(1) # [bs,4]
		for i in range(bs):
			for k in range(channels):
				E[i][i][k] = E_D[i][k] if E_D[i][k]>self.thresold else 1
		return tf.convert_to_tensor(E,dtype=tf.float32)
		
	def call(self,x,state):
		# x: [bs,5] bs the number of persons
		# 5: [x,y,vx,vy,a]
		bs = x.shape[0]
		E = self.get_edge(x)
		x = self.egcn_block(x,E)
		x = tf.concat((x,state),1) # [bs,64+state_dim]
		
		le_h,le_c = self.init_hidden_lstm(bs, self.le_hs)
		ld_h,ld_c = self.init_hidden_lstm(bs, self.ld_hs)
		le_h,_ = self.lstm_encoder(x,(le_h,le_c))
		x,_ = self.lstm_decoder(le_h,(ld_h,ld_c))
		x = activations.relu(x)
		
		x = activations.relu(self.mlps(x))
		
		return x
		
"""
f_gcn = [5,16,32,64]
f_atten = [16,16,16]
n_units = [256,128,64,32,8,2]
le_hs = 256
ld_hs = 256
state_dim =64
		
model = ScaleNet(f_gcn,f_atten,n_units,le_hs,ld_hs,state_dim)
x = tf.random.normal([17,5])
state = tf.random.normal([17,64])

ou = model(x,state)
print(ou.shape)
"""
