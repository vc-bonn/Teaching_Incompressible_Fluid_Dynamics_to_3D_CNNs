from get_param import params,get_hyperparam
import matplotlib.pyplot as plt
from Logger import Logger,t_step
from pde_cnn import get_Net
import torch
import numpy as np
from setups import Dataset
import derivatives as d
from derivatives import vector2HSV,rot_mac,toCuda,toCpu
from torch.optim import Adam
import cv2
import math
import numpy as np
import time
import os
from datetime import datetime
from numpy2vtk import imageToVTK

torch.manual_seed(1)
torch.set_num_threads(4)
np.random.seed(2)


# initialize logger
logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)

# initialize fluid model
pde_cnn = toCuda(get_Net(params))

# load fluid model
date_time,index = logger.load_state(pde_cnn,None,datetime=params.load_date_time,index=params.load_index)
pde_cnn.eval()

print(f"loaded date_time: {date_time}; index: {index}")
model_parameters = filter(lambda p: p.requires_grad, pde_cnn.parameters())
model_parameters = sum([np.prod(p.size()) for p in model_parameters])
print(f"n_model_parameters: {model_parameters}")

# plot color legend
cv2.namedWindow('legend',cv2.WINDOW_NORMAL)
vector = toCuda(torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]))
image = vector2HSV(vector)
image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow('legend',image)

# initialize windows for averaged velocity / pressure fields
cv2.namedWindow('v xy',cv2.WINDOW_NORMAL)
cv2.namedWindow('v xz',cv2.WINDOW_NORMAL)
cv2.namedWindow('v yz',cv2.WINDOW_NORMAL)
cv2.namedWindow('p xy',cv2.WINDOW_NORMAL)
cv2.namedWindow('p xz',cv2.WINDOW_NORMAL)
cv2.namedWindow('p yz',cv2.WINDOW_NORMAL)

def mousePosition_xy(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
		dataset.mousex = y
		dataset.mousey = x

def mousePosition_xz(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
		dataset.mousex = y
		dataset.mousez = x

def mousePosition_yz(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
		dataset.mousey = y
		dataset.mousez = x

cv2.setMouseCallback("v xy",mousePosition_xy)
cv2.setMouseCallback("v xz",mousePosition_xz)
cv2.setMouseCallback("v yz",mousePosition_yz)
cv2.setMouseCallback("p xy",mousePosition_xy)
cv2.setMouseCallback("p xz",mousePosition_xz)
cv2.setMouseCallback("p yz",mousePosition_yz)


last_FPS = 0
quit = False
vtk_iteration = 0
p_pressed = False
recording = False
animation_iteration = 0.0
w_pressed = False
animating = False # animate mu / rho
animation_freq = 45
animation_type = "sin"#"triangle"#
aggregator = "mean"# "max"#

with torch.no_grad():
	while True:
		
		# initialize dataset
		dataset = Dataset(params.width,params.height,params.depth,1,1,interactive=True,average_sequence_length=params.average_sequence_length,max_speed=params.max_speed,dt=params.dt,types=["image","magnus_y","moving_rod_y","ball"],images=["fish","cyber","3_objects"],mu_range=[params.mu_min,params.mu_max],rho_range=[params.rho_min,params.rho_max])
		# options for setup types: "magnus_y","magnus_z","no_box","rod_y","rod_z","moving_rod_y","moving_rod_z","box","benchmark","image","ball"
		# options for images: "submarine","fish","cyber","wing","2_objects","3_objects"
		
		FPS=0
		last_time = time.time()
		
		# Start: fluid simulation loop
		
		for t in range(params.average_sequence_length):
			
			# get dirichlet boundary conditions, fluid domain, vector potential (streamfunction), pressure field, mu, rho from dataset:
			v_cond,cond_mask,a_old,p_old,mu,rho = toCuda(dataset.ask())
			
			v_cond = d.normal2staggered(v_cond) # map dirichlet boundary conditions onto staggered grid
			
			a_new,p_new = pde_cnn(a_old,p_old,v_cond,cond_mask,mu,rho) # apply fluid model on fluid state and boundary conditions
			
			dataset.tell(toCpu(a_new),toCpu(p_new)) # update dataset with new predicted fluid state
			
			# End: fluid simulation loop
			
			# Visualization Code:
			if t%1==0:
				print(f"t:{t}")
				
				cond_mask_mac = (d.normal2staggered(cond_mask.repeat(1,3,1,1,1))==1).float()
				flow_mask_mac = 1-cond_mask_mac
				
				# show velocity field
				
				v_new = d.staggered2normal(flow_mask_mac*d.rot_mac(a_new)+cond_mask_mac*v_cond)[:,:,1:-1,1:-1,1:-1]
				
				if aggregator == "mean":
					vector = (v_new[0])[(0,1),].mean(3).clone()
				elif aggregator == "max":
					vector = torch.max((v_new[0])[(0,1),],dim=3)[0].clone()
				image = vector2HSV(vector)
				image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
				cv2.imshow('v xy',image)
				
				if aggregator == "mean":
					vector = (v_new[0])[(0,2),].mean(2).clone()
				if aggregator == "max":
					vector = torch.max((v_new[0])[(0,2),],dim=2)[0].clone()
				image = vector2HSV(vector)
				image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
				cv2.imshow('v xz',image)
				
				if aggregator == "mean":
					vector = (v_new[0])[(1,2),].mean(1).clone()
				if aggregator == "max":
					vector = torch.max((v_new[0])[(1,2),],dim=1)[0].clone()
				image = vector2HSV(vector)
				image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
				cv2.imshow('v yz',image)
				
				# show pressure field
				
				if aggregator == "mean":
					p = (p_new[0,0]*(1-cond_mask[0,0])).mean(2).clone()
				elif aggregator == "max":
					p = torch.max((p_new[0,0]*(1-cond_mask[0,0])),dim=2)[0].clone()
				p = p-torch.min(p)
				p = p/torch.max(p)
				p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy()
				cv2.imshow('p xy',p)
				
				if aggregator == "mean":
					p = (p_new[0,0]*(1-cond_mask[0,0])).mean(1).clone()
				elif aggregator == "max":
					p = torch.max((p_new[0,0]*(1-cond_mask[0,0])),dim=1)[0].clone()
				p = p-torch.min(p)
				p = p/torch.max(p)
				p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy()
				cv2.imshow('p xz',p)
				
				if aggregator == "mean":
					p = (p_new[0,0]*(1-cond_mask[0,0])).mean(0).clone()
				elif aggregator == "max":
					p = torch.max((p_new[0,0]*(1-cond_mask[0,0])),dim=0)[0].clone()
				p = p-torch.min(p)
				p = p/torch.max(p)
				p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy()
				cv2.imshow('p yz',p)
				
				divergence_v = d.div(flow_mask_mac*d.rot_mac(a_new)+cond_mask_mac*v_cond)[:,:,1:-1,1:-1,1:-1]
				
				print(f"FPS: {last_FPS}; E[|div(v)|] = {torch.mean(torch.abs(divergence_v))}; E[div(v)^2] = {torch.mean(divergence_v**2)}")
				print(f"mu: {dataset.mousemu.numpy()[0,0,0,0]}; rho: {dataset.mouserho.numpy()[0,0,0,0]}; v: {dataset.mousev}")
				
				key = cv2.waitKey(1)
				
				if key==ord('x'):
					dataset.mousev+=0.1
				elif key==ord('y'):
					dataset.mousev-=0.1
				
				if key==ord('s'):
					dataset.mousew+=0.1
				elif key==ord('a'):
					dataset.mousew-=0.1
				
				if key==ord('f'):
					dataset.mousemu*=1.05
				elif key==ord('d'):
					dataset.mousemu/=1.05
				
				if key==ord('v'):
					dataset.mouserho*=1.05
				elif key==ord('c'):
					dataset.mouserho/=1.05
					
				elif key==ord('1'):
					dataset.mousemu=5
					dataset.mouserho=0.2
					dataset.mousev=-1
				elif key==ord('2'):
					dataset.mousemu=0.5
					dataset.mouserho=1
					dataset.mousev=-1
				elif key==ord('3'):
					dataset.mousemu=0.2
					dataset.mouserho=1
					dataset.mousev=-1
				elif key==ord('4'):
					dataset.mousemu=0.1
					dataset.mouserho=5
					dataset.mousev=-1
				elif key==ord('5'):
					dataset.mousemu=0.02
					dataset.mouserho=10
					dataset.mousev=-1
				
				if key==ord('r'):
					if dataset.env_info[0]["type"] == "image":
						dataset.mousex=64
						dataset.mousey=32
						dataset.mousez=32
						dataset.mousev=-1
						dataset.mousew=0
					
					if dataset.env_info[0]["type"] == "magnus_y":
						dataset.mousex=100
						dataset.mousey=32
						dataset.mousez=32
						dataset.mousev=-1
						dataset.mousew=1
				
				# animate mu / rho (for movie)
				if key==ord('w') or animating:
					animation_time = animation_iteration/15
					# triangle animation (value between 0 and 1)
					if animation_type == "triangle":
						value = animation_time % animation_freq
						if value>animation_freq/2:
							value = animation_freq-value
						value /= animation_freq/2
					elif animation_type == "sin":
						value = np.sin(animation_time/animation_freq*2*np.pi)/2+0.5
					
					dataset.mousemu = np.exp(value*(np.log(5)-np.log(0.01))+np.log(0.01))
					dataset.mouserho = np.exp(value*(np.log(0.2)-np.log(8))+np.log(8))
					
					animation_iteration += 1
					if not w_pressed and key==ord('w'):
						animating = not animating
					w_pressed = True
				if key != ord('w'):
					w_pressed = False
				
				# print to VTK
				if key==ord('p') or recording:
					name = dataset.env_info[0]["type"]
					if name=="image":
						name = name+"_"+dataset.env_info[0]["image"]
					os.makedirs(f"vtk/{name}/{get_hyperparam(params)}",exist_ok=True)
					
					pressure = toCpu((p_new[0,0]*(1-cond_mask[0,0])).clone()).numpy()
					imageToVTK(f"vtk/{name}/{get_hyperparam(params)}/pressure.{vtk_iteration}",pointData={"pressure":pressure})
					v_new = toCpu(d.staggered2normal(flow_mask_mac*d.rot_mac(a_new)+cond_mask_mac*v_cond)[:,:,1:-1,1:-1,1:-1])
					imageToVTK(f"vtk/{name}/{get_hyperparam(params)}/velocity.{vtk_iteration}",cellData={"velocity":(v_new[0,0].numpy(),v_new[0,1].numpy(),v_new[0,2].numpy())})
					boundary_object = cond_mask[0,0].clone()
					imageToVTK(f"vtk/{name}/{get_hyperparam(params)}/boundary.{vtk_iteration}",pointData={"boundary":toCpu(1-boundary_object).numpy()})
					print(f"saved fluid state to vtk-files in: vtk/{name}/{get_hyperparam(params)} ({vtk_iteration})")
					vtk_iteration += 1
					if not p_pressed and key==ord('p'):
						recording = not recording
					p_pressed = True
				if key != ord('p'):
					p_pressed = False
				
				if key==ord('n'):
					break
				
				if key==ord('q'):
					quit=True
					break
				
				FPS += 1
				if time.time()-last_time>=1:
					last_time = time.time()
					last_FPS=FPS
					FPS = 0
		
		if quit:
			break

cv2.destroy_all_windows()
