import torch
import torch.nn.functional as F
import math
from get_param import params,toCuda,toCpu

# dx/dy/dz "centered"

dx_kernel = toCuda(torch.Tensor([-0.5,0,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4))
def dx(v):
	return F.conv3d(v,dx_kernel,padding=(1,0,0))

dy_kernel = toCuda(torch.Tensor([-0.5,0,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4))
def dy(v):
	return F.conv3d(v,dy_kernel,padding=(0,1,0))

dz_kernel = toCuda(torch.Tensor([-0.5,0,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3))
def dz(v):
	return F.conv3d(v,dz_kernel,padding=(0,0,1))

# dx/dy/dz "minus" half a voxel shifted

dx_m_kernel = toCuda(torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4))
def dx_m(v):
	return F.conv3d(v,dx_m_kernel,padding=(1,0,0))

dy_m_kernel = toCuda(torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4))
def dy_m(v):
	return F.conv3d(v,dy_m_kernel,padding=(0,1,0))

dz_m_kernel = toCuda(torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3))
def dz_m(v):
	return F.conv3d(v,dz_m_kernel,padding=(0,0,1))

# dx/dy/dz "plus" half a voxel shifted

dx_p_kernel = toCuda(torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4))
def dx_p(v):
	return F.conv3d(v,dx_p_kernel,padding=(1,0,0))

dy_p_kernel = toCuda(torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4))
def dy_p(v):
	return F.conv3d(v,dy_p_kernel,padding=(0,1,0))

dz_p_kernel = toCuda(torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3))
def dz_p(v):
	return F.conv3d(v,dz_p_kernel,padding=(0,0,1))


# mean x/y/z "minus" half a voxel shifted

mean_x_m_kernel = toCuda(torch.Tensor([0.5,0.5,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4))
def mean_x_m(v):
	return F.conv3d(v,mean_x_m_kernel,padding=(1,0,0))

mean_y_m_kernel = toCuda(torch.Tensor([0.5,0.5,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4))
def mean_y_m(v):
	return F.conv3d(v,mean_y_m_kernel,padding=(0,1,0))

mean_z_m_kernel = toCuda(torch.Tensor([0.5,0.5,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3))
def mean_z_m(v):
	return F.conv3d(v,mean_z_m_kernel,padding=(0,0,1))

# mean x/y/z "plus" half a voxel shifted

mean_x_p_kernel = toCuda(torch.Tensor([0,0.5,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4))
def mean_x_p(v):
	return F.conv3d(v,mean_x_p_kernel,padding=(1,0,0))

mean_y_p_kernel = toCuda(torch.Tensor([0,0.5,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4))
def mean_y_p(v):
	return F.conv3d(v,mean_y_p_kernel,padding=(0,1,0))

mean_z_p_kernel = toCuda(torch.Tensor([0,0.5,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3))
def mean_z_p(v):
	return F.conv3d(v,mean_z_p_kernel,padding=(0,0,1))


def rot_mac(a):
	return torch.cat([dy_p(a[:,2:3])-dz_p(a[:,1:2]),dz_p(a[:,0:1])-dx_p(a[:,2:3]),dx_p(a[:,1:2])-dy_p(a[:,0:1])],dim=1)

def div(v):
	return dx_p(v[:,0:1])+dy_p(v[:,1:2])+dz_p(v[:,2:3])


# map velocities (to vx)
map_vy2vx_p_kernel = toCuda(torch.Tensor([[0,0,0.5],[0,0,0.5],[0,0,0]]).unsqueeze(0).unsqueeze(1).unsqueeze(4))
def map_vy2vx_p(v):
	return F.conv3d(v,map_vy2vx_p_kernel,padding=(1,1,0))

def map_vy2vx_m(v):
	return mean_x_m(v)

map_vz2vx_p_kernel = toCuda(torch.Tensor([[0,0,0.5],[0,0,0.5],[0,0,0]]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
def map_vz2vx_p(v):
	return F.conv3d(v,map_vz2vx_p_kernel,padding=(1,0,1))

def map_vz2vx_m(v):
	return mean_x_m(v)

# map velocities (to vy)
map_vx2vy_p_kernel = toCuda(torch.Tensor([[0,0,0],[0,0,0],[0.5,0.5,0]]).unsqueeze(0).unsqueeze(1).unsqueeze(4))
def map_vx2vy_p(v):
	return F.conv3d(v,map_vx2vy_p_kernel,padding=(1,1,0))

def map_vx2vy_m(v):
	return mean_y_m(v)

map_vz2vy_p_kernel = toCuda(torch.Tensor([[0,0,0.5],[0,0,0.5],[0,0,0]]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
def map_vz2vy_p(v):
	return F.conv3d(v,map_vz2vy_p_kernel,padding=(0,1,1))

def map_vz2vy_m(v):
	return mean_y_m(v)

# map velocities (to vz)
map_vx2vz_p_kernel = toCuda(torch.Tensor([[0,0,0],[0,0,0],[0.5,0.5,0]]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
def map_vx2vz_p(v):
	return F.conv3d(v,map_vx2vz_p_kernel,padding=(1,0,1))

def map_vx2vz_m(v):
	return mean_z_m(v)

map_vy2vz_p_kernel = toCuda(torch.Tensor([[0,0,0],[0,0,0],[0.5,0.5,0]]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
def map_vy2vz_p(v):
	return F.conv3d(v,map_vy2vz_p_kernel,padding=(0,1,1))

def map_vy2vz_m(v):
	return mean_z_m(v)



#laplace_kernel = toCuda(torch.Tensor([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,-6,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]]).unsqueeze(0).unsqueeze(1))# 7 point stencil
laplace_kernel = toCuda(1/26*torch.Tensor([[[2,3,2],[3,6,3],[2,3,2]],[[3,6,3],[6,-88,6],[3,6,3]],[[2,3,2],[3,6,3],[2,3,2]]]).unsqueeze(0).unsqueeze(1))# 27 point stencil
def laplace(v):
	return F.conv3d(v,laplace_kernel,padding=(1,1,1))


# staggered: MAC grid
# normal: v & p share the same coordinates

def staggered2normal(v):
	v[:,0:1] = mean_x_p(v[:,0:1])
	v[:,1:2] = mean_y_p(v[:,1:2])
	v[:,2:3] = mean_z_p(v[:,2:3])
	return v

def normal2staggered(v):
	v[:,0:1] = mean_x_m(v[:,0:1])
	v[:,1:2] = mean_y_m(v[:,1:2])
	v[:,2:3] = mean_z_m(v[:,2:3])
	return v


size = 2
border_kernel_x = toCuda(torch.zeros(3,3,2*size+1,1,1))
border_kernel_y = toCuda(torch.zeros(3,3,1,2*size+1,1))
border_kernel_z = toCuda(torch.zeros(3,3,1,1,2*size+1))
for i in range(3):
	border_kernel_x[i,i,:,:,:] = 1/(2*size+1)
	border_kernel_y[i,i,:,:,:] = 1/(2*size+1)
	border_kernel_z[i,i,:,:,:] = 1/(2*size+1)

def get_borders(boundary):
	"""
	:boundary: domain boundary (batch_size x 3 x w x h x d)
	:return: borders of domain boundary (batch_size x 3 x w x h x d)
	"""
	border = boundary
	border = F.conv3d(border,border_kernel_x,padding=(size,0,0))
	border = F.conv3d(border,border_kernel_y,padding=(0,size,0))
	border = F.conv3d(border,border_kernel_z,padding=(0,0,size))
	
	border = boundary*((border<0.99).float())
	return border

def vector2HSV(vector,plot_sqrt=False):
	"""
	:vector: vector field (size: 2 x height x width)
	:return: hsv (hue: direction of vector; saturation: 1; value: abs value of vector)
	"""
	values = torch.sqrt(torch.sum(torch.pow(vector,2),dim=0)).unsqueeze(0)
	saturation = toCuda(torch.ones(values.shape))
	norm = vector/(values+0.000001)
	angles = torch.asin(norm[0])+math.pi/2
	angles[norm[1]<0] = 2*math.pi-angles[norm[1]<0]
	hue = angles.unsqueeze(0)/(2*math.pi)
	hue = (hue*360+100)%360
	#values = norm*torch.log(values+1)
	values = values/torch.max(values)
	if plot_sqrt:
		values = torch.sqrt(values)
	hsv = torch.cat([hue,saturation,values])
	return hsv.permute(1,2,0).cpu().numpy()
