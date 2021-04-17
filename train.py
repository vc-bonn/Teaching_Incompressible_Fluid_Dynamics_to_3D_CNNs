from get_param import params,get_hyperparam
import torch
from torch.optim import Adam
import numpy as np
from derivatives import toCuda,toCpu
import derivatives as d
from derivatives import vector2HSV,rot_mac
from setups import Dataset
from Logger import Logger,t_step
from pde_cnn import get_Net

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

# initialize model
pde_cnn = toCuda(get_Net(params))
pde_cnn.train()

# initialize optimizer
optimizer = Adam(pde_cnn.parameters(),lr=params.lr)

# initialize logger and, if demanded, load previous model / optimizer
logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=params.log)
if params.load_latest or params.load_date_time is not None or params.load_index is not None:
	load_logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)
	if params.load_optimizer:
		params.load_date_time, params.load_index = logger.load_state(pde_cnn,optimizer,params.load_date_time,params.load_index)
	else:
		params.load_date_time, params.load_index = logger.load_state(pde_cnn,None,params.load_date_time,params.load_index)
	params.load_index=int(params.load_index)
	print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

# initialize dataset
dataset = Dataset(params.width,params.height,params.depth,params.batch_size,params.dataset_size,params.average_sequence_length,max_speed=params.max_speed,dt=params.dt,types=["box","moving_rod_y","moving_rod_z","magnus_y","magnus_z","ball"],mu_range=[params.mu_min,params.mu_max],rho_range=[params.rho_min,params.rho_max])

eps = 0.00000001

def loss_function(x):
	if params.loss=="square":
		return torch.pow(x,2)
	if params.loss=="exp_square":
		x = torch.pow(x,2)
		return torch.exp(x/torch.max(x).detach()*5)
	if params.loss=="abs":
		return torch.abs(x)
	if params.loss=="log_square":
		return torch.log(torch.pow(x,2)+eps)

for epoch in range(params.load_index,params.n_epochs):

	for i in range(params.n_batches_per_epoch):
		# draw batch from dataset
		v_cond,cond_mask,a_old,p_old,mu,rho = toCuda(dataset.ask())
		
		# map v_cond to MAC grid
		v_cond = d.normal2staggered(v_cond)
		
		# apply fluid model on fluid state / boundary conditions for given mu and rho
		a_new,p_new = pde_cnn(a_old,p_old,v_cond,cond_mask,mu,rho)
		v_new = d.rot_mac(a_new)
		
		# compute masks for fluid domain / boundary conditions and map them to MAC grid:
		cond_mask_mac = (d.normal2staggered(cond_mask.repeat(1,3,1,1,1))==1).float()
		flow_mask_mac = 1-cond_mask_mac
		
		#weight cond_mask_mac stronger at domain borders:
		cond_mask_mac = cond_mask_mac + params.loss_border * d.get_borders(cond_mask_mac)
		
		# compute loss on domain boundaries
		loss_bound = torch.mean(loss_function(cond_mask_mac*(v_new-v_cond))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))
		
		# compute loss for Navier Stokes equations
		v_old = d.rot_mac(a_old)
		
		if params.integrator == "explicit":
			v = v_old
		if params.integrator == "implicit":
			v = v_new
		if params.integrator == "imex":
			v = (v_new+v_old)/2
		
		loss_nav =  torch.mean(loss_function(flow_mask_mac*(rho*((v_new[:,0:1]-v_old[:,0:1])/params.dt+v[:,0:1]*d.dx(v[:,0:1])+0.5*(d.map_vy2vx_p(v[:,1:2])*d.dy_p(v[:,0:1])+d.map_vy2vx_m(v[:,1:2])*d.dy_m(v[:,0:1]))+0.5*(d.map_vz2vx_p(v[:,2:3])*d.dz_p(v[:,0:1])+d.map_vz2vx_m(v[:,2:3])*d.dz_m(v[:,0:1])))+d.dx_m(p_new)-mu*d.laplace(v[:,0:1])))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))+\
					torch.mean(loss_function(flow_mask_mac*(rho*((v_new[:,1:2]-v_old[:,1:2])/params.dt+v[:,1:2]*d.dy(v[:,1:2])+0.5*(d.map_vx2vy_p(v[:,0:1])*d.dx_p(v[:,1:2])+d.map_vx2vy_m(v[:,0:1])*d.dx_m(v[:,1:2]))+0.5*(d.map_vz2vy_p(v[:,2:3])*d.dz_p(v[:,1:2])+d.map_vz2vy_m(v[:,2:3])*d.dz_m(v[:,1:2])))+d.dy_m(p_new)-mu*d.laplace(v[:,1:2])))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))+\
					torch.mean(loss_function(flow_mask_mac*(rho*((v_new[:,2:3]-v_old[:,2:3])/params.dt+v[:,2:3]*d.dz(v[:,2:3])+0.5*(d.map_vx2vz_p(v[:,0:1])*d.dx_p(v[:,2:3])+d.map_vx2vz_m(v[:,0:1])*d.dx_m(v[:,2:3]))+0.5*(d.map_vy2vz_p(v[:,1:2])*d.dy_p(v[:,2:3])+d.map_vy2vz_m(v[:,1:2])*d.dy_m(v[:,2:3])))+d.dz_m(p_new)-mu*d.laplace(v[:,2:3])))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))
		
		# combine loss terms for boundary conditions / Navier Stokes equations
		loss = params.loss_bound*loss_bound + params.loss_nav*loss_nav
		
		# evt put some extra loss on the mean of the vector potential
		if params.loss_mean_a != 0:
			loss_mean_a = torch.mean(a_new,dim=(1,2,3,4))**2
			loss = loss + params.loss_mean_a*loss_mean_a
		
		# evt put some extra loss on the mean of the pressure field
		if params.loss_mean_p != 0:
			loss_mean_p = torch.mean(p_new,dim=(1,2,3,4))**2
			loss = loss + params.loss_mean_p*loss_mean_p
		
		# evt regularize gradient of pressure field (might be useful for very high Reynolds numbers)
		if params.regularize_grad_p != 0:
			regularize_grad_p = torch.mean((dx_right(p_new)**2+dy_bottom(p_new)**2)[:,:,2:-2,2:-2,2:-2],dim=(1,2,3,4))
			loss = loss + params.regularize_grad_p*regularize_grad_p
		
		if params.loss == "log_square" or params.loss == "exp_square":
			loss = torch.mean(loss)
		elif params.loss=='square' or params.loss=='abs':
			loss = torch.mean(torch.log(loss))
		
		# compute gradients for model parameters
		optimizer.zero_grad()
		loss = loss*params.loss_multiplier
		loss.backward()
		
		# evt clip gradients
		if params.clip_grad_value is not None:
			torch.nn.utils.clip_grad_value_(pde_cnn.parameters(),3*params.clip_grad_value)
		
		if params.clip_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(pde_cnn.parameters(),params.clip_grad_norm)
		
		# perform optimization step on model
		optimizer.step()
		
		# update dataset with predicted fluid state in order to fill up dataset with more and more realistic fluid states
		dataset.tell(toCpu(a_new),toCpu(p_new))
		
		# log losses
		loss = toCpu(loss).numpy()
		loss_bound = toCpu(torch.mean(loss_bound)).numpy()
		loss_nav = toCpu(torch.mean(loss_nav)).numpy()
		
		if i%1 == 0:
			logger.log(f"loss_{params.loss}",loss,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_bound_{params.loss}",loss_bound,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_nav_{params.loss}",loss_nav,epoch*params.n_batches_per_epoch+i)
			
			if params.loss_mean_a != 0:
				loss_mean_a = toCpu(torch.mean(loss_mean_a)).numpy()
				logger.log(f"loss_mean_a",loss_mean_a,epoch*params.n_batches_per_epoch+i)
			
			if params.loss_mean_p != 0:
				loss_mean_p = toCpu(torch.mean(loss_mean_p)).numpy()
				logger.log(f"loss_mean_p",loss_mean_p,epoch*params.n_batches_per_epoch+i)
			
			if params.regularize_grad_p != 0:
				regularize_grad_p = toCpu(torch.mean(regularize_grad_p)).numpy()
				logger.log(f"regularize_grad_p",regularize_grad_p,epoch*params.n_batches_per_epoch+i)
		
		if i%1 == 0:
			print(f"{epoch}: i:{i}: loss: {loss}; loss_bound: {loss_bound}; loss_nav: {loss_nav};")
	
	# save model / optimizer states
	if params.log:
		logger.save_state(pde_cnn,optimizer,epoch+1)
