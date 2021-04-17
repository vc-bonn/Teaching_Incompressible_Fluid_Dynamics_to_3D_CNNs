import argparse

def str2bool(v):
	"""
	'type variable' for add_argument
	"""
	if v.lower() in ('yes','true','t','y','1'):
		return True
	elif v.lower() in ('no','false','f','n','0'):
		return False
	else:
		raise argparse.ArgumentTypeError('boolean value expected.')

def params():
	"""
	return parameters for training / testing / plotting of models
	:return: parameter-Namespace
	"""
	parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')
	
	# Training parameters
	parser.add_argument('--net', default="UNet", type=str, help='network to train', choices=["UNet","pruned_UNet"])
	parser.add_argument('--n_epochs', default=1500, type=int, help='number of epochs (after each epoch, the model gets saved)')
	parser.add_argument('--hidden_size', default=15, type=int, help='hidden size of network (default: 15)')
	parser.add_argument('--n_batches_per_epoch', default=1000, type=int, help='number of batches per epoch (default: 1000)')
	parser.add_argument('--batch_size', default=10, type=int, help='batch size (default: 10)')
	parser.add_argument('--average_sequence_length', default=5000, type=int, help='average sequence length in dataset (default: 5000)')
	parser.add_argument('--dataset_size', default=1000, type=int, help='size of dataset (default: 1000)')
	parser.add_argument('--cuda', default=True, type=str2bool, help='use GPU')
	parser.add_argument('--loss_bound', default=1, type=float, help='loss factor for boundary conditions')
	parser.add_argument('--loss_border', default=20, type=float, help='loss factor for extra weight on boundary borders')
	parser.add_argument('--loss_cont', default=50, type=float, help='loss factor for continuity equation')
	parser.add_argument('--loss_nav', default=1, type=float, help='loss factor for navier stokes equations')
	parser.add_argument('--loss_rho', default=10, type=float, help='loss factor for keeping rho fixed')
	parser.add_argument('--loss_mean_a', default=0, type=float, help='loss factor to keep mean of a around 0')
	parser.add_argument('--loss_mean_p', default=0, type=float, help='loss factor to keep mean of p around 0')
	parser.add_argument('--regularize_grad_p', default=0, type=float, help='regularizer for gradient of p. evt needed for very high reynolds numbers (default: 0)')
	parser.add_argument('--max_speed', default=1, type=float, help='max speed for boundary conditions in dataset (default: 1)')
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer (default: 0.001)')
	parser.add_argument('--lr_grad', default=0.001, type=float, help='learning rate of optimizer (default: 0.001)')
	parser.add_argument('--clip_grad_norm', default=None, type=float, help='gradient norm clipping (default: None)')
	parser.add_argument('--clip_grad_value', default=None, type=float, help='gradient value clipping (default: None)')
	parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training (turn off for debugging)')
	parser.add_argument('--log_grad', default=False, type=str2bool, help='log gradients during training (turn on for debugging)')
	parser.add_argument('--plot_sqrt', default=False, type=str2bool, help='plot sqrt of velocity value (to better distinguish directions at low velocities)')
	parser.add_argument('--plot', default=False, type=str2bool, help='plot during training')
	parser.add_argument('--flip', default=False, type=str2bool, help='flip training samples randomly during training (default: False)')
	parser.add_argument('--integrator', default='imex', type=str, help='integration scheme (explicit / implicit / imex) (default: imex)',choices=['explicit','implicit','imex'])
	parser.add_argument('--loss', default='square', type=str, help='loss type to train network (default: square)',choices=['square','abs','log_square','exp_square'])
	parser.add_argument('--loss_multiplier', default=1, type=float, help='multiply loss / gradients (default: 1)')
	
	# Setup parameters
	parser.add_argument('--width', default=128, type=int, help='setup width')
	parser.add_argument('--height', default=64, type=int, help='setup height')
	parser.add_argument('--depth', default=64, type=int, help='setup depth')
	
	# Fluid parameters
	parser.add_argument('--rho_min', default=0.1, type=float, help='min of fluid density rho (default 0.1)')
	parser.add_argument('--rho_max', default=15, type=float, help='max of fluid density rho (default 15)')
	parser.add_argument('--mu_min', default=0.01, type=float, help='min of fluid viscosity mu (default 0.01)')
	parser.add_argument('--mu_max', default=8, type=float, help='max of fluid viscosity mu (default 8)')
	parser.add_argument('--dt', default=1, type=float, help='timestep of fluid integrator')
	
	# Load parameters
	parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
	parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
	parser.add_argument('--load_optimizer', default=False, type=str2bool, help='load state of optimizer (default: True)')
	parser.add_argument('--load_latest', default=False, type=str2bool, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
	
	# parse parameters
	params = parser.parse_args()
	
	return params

def get_hyperparam(params):
	return f"net {params.net}; hs {params.hidden_size}; dt {params.dt};"

params = params()

def toCuda(x):
	if type(x) is tuple:
		return [xi.cuda() if params.cuda else xi for xi in x]
	return x.cuda() if params.cuda else x

def toCpu(x):
	if type(x) is tuple:
		return [xi.detach().cpu() for xi in x]
	return x.detach().cpu()
