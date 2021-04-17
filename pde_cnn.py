import torch
from torch import nn
from derivatives import rot_mac
import torch.nn.functional as F
from derivatives import toCuda,toCpu

def get_Net(params):
	if params.net == "UNet":
		pde_cnn = PDE_UNet(params.hidden_size)
	if params.net == "pruned_UNet":
		pde_cnn = PDE_pruned_UNet(params.hidden_size)
	return pde_cnn

def lrelu(x,alpha=0.001):
	return torch.relu(x)+alpha*x

def conv_block_3d(in_dim, out_dim, activation):
	return nn.Sequential(
		nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
		nn.BatchNorm3d(out_dim),
		activation,)

def conv_trans_block_3d(in_dim, out_dim, activation):
	return nn.Sequential(
		nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
		nn.BatchNorm3d(out_dim),
		activation,)

def max_pooling_3d():
	return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

def conv_block_2_3d(in_dim, out_dim, activation):
	return nn.Sequential(
		conv_block_3d(in_dim, out_dim, activation),
		nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
		nn.BatchNorm3d(out_dim),)

class PDE_UNet(nn.Module):
	
	def __init__(self, hidden_size):
		super(PDE_UNet, self).__init__()
		
		self.in_dim = 16+2
		self.out_dim = 4
		self.num_filters = hidden_size
		activation = nn.LeakyReLU(0.2, inplace=True)
		
		# Down sampling
		self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
		self.pool_1 = max_pooling_3d()
		self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
		self.pool_2 = max_pooling_3d()
		self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
		self.pool_3 = max_pooling_3d()
		self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
		self.pool_4 = max_pooling_3d()
		self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
		self.pool_5 = max_pooling_3d()
		
		# Bridge
		self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
		
		# Up sampling
		self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
		self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
		self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
		self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
		self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
		self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
		self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
		self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
		self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
		self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
		
		# Output
		#self.out = conv_block_3d(self.num_filters, self.out_dim, activation) #This is probably a bug!
		self.out = nn.Conv3d(self.num_filters, self.out_dim, kernel_size=3, stride=1, padding=1)
		self.out_bn = nn.BatchNorm3d(self.out_dim)

	def forward(self,a_old,p_old,v_cond,mask_cond,mu,rho):
		mask_flow = 1-mask_cond
		v_old = rot_mac(a_old)#this could be probably learned rather easily (evtl not needed)
		ones = toCuda(torch.ones(p_old.shape))
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,torch.log(mu)*ones,torch.log(rho)*ones],dim=1)
		#1+3+3+1+3+1+1+3=16
		
		# Down sampling
		down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
		pool_1 = self.pool_1(down_1) # -> [1, 4, 64, 64, 64]
		
		down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
		pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32]
		
		down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
		pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16]
		
		down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
		pool_4 = self.pool_4(down_4) # -> [1, 32, 8, 8, 8]
		
		down_5 = self.down_5(pool_4) # -> [1, 64, 8, 8, 8]
		pool_5 = self.pool_5(down_5) # -> [1, 64, 4, 4, 4]
		
		# Bridge
		bridge = self.bridge(pool_5) # -> [1, 128, 4, 4, 4]
		
		# Up sampling
		trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
		concat_1 = torch.cat([trans_1, down_5], dim=1) # -> [1, 192, 8, 8, 8]
		up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
		
		trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
		concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 16, 16, 16]
		up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
		
		trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
		concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 32, 32, 32]
		up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
		
		trans_4 = self.trans_4(up_3) # -> [1, 16, 64, 64, 64]
		concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 64, 64, 64]
		up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]
		
		trans_5 = self.trans_5(up_4) # -> [1, 8, 128, 128, 128]
		concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 128, 128, 128]
		up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]
		
		# Output
		out = self.out_bn(self.out(up_5)) # -> [1, 4, 128, 128, 128]
		
		a_new, p_new = 400*torch.tanh((a_old+out[:,0:3])/400), 10*torch.tanh((p_old+out[:,3:4])/10)
		
		p_new.data = (p_new.data-torch.mean(p_new.data,dim=(1,2,3,4)).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)) # normalize pressure # This should be part of the model!
		a_new.data = (a_new.data-torch.mean(a_new.data,dim=(2,3,4)).unsqueeze(2).unsqueeze(3).unsqueeze(4)) # normalize a
		
		return a_new,p_new


class PDE_pruned_UNet(nn.Module):
	
	def __init__(self, hidden_size):
		super(PDE_pruned_UNet, self).__init__()
		
		self.num_filters = hidden_size
		activation = nn.LeakyReLU(0.2, inplace=True)
		self.in_dim = 16+2
		self.out_dim = 4
		
		# Down sampling
		self.down_1 = conv_block_3d(self.in_dim, self.num_filters, activation)
		self.pool_1 = max_pooling_3d()
		self.down_2 = conv_block_3d(self.num_filters, self.num_filters*2, activation)
		self.pool_2 = max_pooling_3d()
		self.down_3 = conv_block_3d(self.num_filters*2, self.num_filters*4, activation)
		self.pool_3 = max_pooling_3d()
		
		# Bridge
		self.bridge = conv_block_3d(self.num_filters*4, self.num_filters*8, activation)
		
		# Up sampling
		self.trans_1 = conv_trans_block_3d(self.num_filters*8, self.num_filters*4, activation)
		self.up_1 = conv_block_3d(self.num_filters*4, self.num_filters*4, activation)
		self.trans_2 = conv_trans_block_3d(self.num_filters*4, self.num_filters*2, activation)
		self.up_2 = conv_block_3d(self.num_filters*2, self.num_filters*2, activation)
		self.trans_3 = conv_trans_block_3d(self.num_filters*2, self.num_filters, activation)
		self.up_3 = conv_block_3d(self.num_filters, self.num_filters, activation)
		
		# Output
		self.out = nn.Conv3d(self.num_filters, self.out_dim, kernel_size=3, stride=1, padding=1)
		self.out_bn = nn.BatchNorm3d(self.out_dim)
	
	def forward(self,a_old,p_old,v_cond,mask_cond,mu,rho):
		mask_flow = 1-mask_cond
		v_old = rot_mac(a_old)#this could be probably learned rather easily (evtl not needed)
		ones = toCuda(torch.ones(p_old.shape))
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,torch.log(mu)*ones,torch.log(rho)*ones],dim=1)
		
		# Down sampling
		down_1 = self.down_1(x)
		pool_1 = self.pool_1(down_1)
		
		down_2 = self.down_2(pool_1)
		pool_2 = self.pool_2(down_2)
		
		down_3 = self.down_3(pool_2)
		pool_3 = self.pool_3(down_3)
		
		# Bridge
		bridge = self.bridge(pool_3)
		
		# Up sampling
		trans_1 = self.trans_1(bridge)
		concat_1 = trans_1 + down_3
		up_1 = self.up_1(concat_1)
		
		trans_2 = self.trans_2(up_1)
		concat_2 = trans_2 + down_2
		up_2 = self.up_2(concat_2)
		
		trans_3 = self.trans_3(up_2)
		concat_3 = trans_3 + down_1
		up_3 = self.up_3(concat_3)
		
		# Output
		out = self.out_bn(self.out(up_3))
		
		a_new, p_new = 400*torch.tanh((a_old+out[:,0:3])/400), 10*torch.tanh((p_old+out[:,3:4])/10)
		
		p_new.data = (p_new.data-torch.mean(p_new.data,dim=(1,2,3,4)).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)) # normalize pressure # This should be part of the model!
		a_new.data = (a_new.data-torch.mean(a_new.data,dim=(2,3,4)).unsqueeze(2).unsqueeze(3).unsqueeze(4)) # normalize a
		
		return a_new, p_new
