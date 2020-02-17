from read import *
from functions import *
import sys
import os
import time
import argparse

#create argument parser
parser = argparse.ArgumentParser()

#add arguments
parser.add_argument("-bioassay_list_file", help="Path to bioassay list file")
parser.add_argument("-bioassay_comp_file", help="Path to assay comp file (bioassay and compound appearances)")
parser.add_argument("-bioassay_id", help="Bioassay ID")
parser.add_argument("-alpha", help="Alpha: weight on L_c")
#parser.add_argument("-L_c_option", help="L_c option: '1' -- overall nCI,  '2' -- selective nCI plus non-selective nCI")
parser.add_argument("-beta", help="Beta: weight on L_R")
parser.add_argument("-lrate", help="Learning rate")
parser.add_argument("-epsilon", help="Epsilon")
parser.add_argument("-max_iter", help="Maximum number of iteration")
parser.add_argument("-model_file", help="Path to output model file")
#parser.add_argument("-kernel", help="Kernal to be used: 'linear' -- linear kernel, 'tanimoto' -- tanimoto kernel")
parser.add_argument("-matrix_format", help="Input file format: 'dense' -- dense input matrix, 'sparse' -- sparse input matrix")
parser.add_argument("-loss_file", help="Path to loss file")
parser.add_argument("-theta_plus", help="Theta_plus: scale factor to adjust push-up weight")
parser.add_argument("-XI_plus", help="XI_plus: push-up threshold")
parser.add_argument("-theta_minus", help="Theta_minus: scale factor to adjust push-down weight")
parser.add_argument("-XI_minus", help="XI_minus: push-down threshold")

#parse arguments
args = parser.parse_args()


#convert arguments to variables
bioassay_list_file	= str(args.bioassay_list_file)
bioassay_comp_file	= str(args.bioassay_comp_file)
bioassay_id             = str(args.bioassay_id)
alpha                   = float(args.alpha)
L_c_option              = "1"
beta                    = float(args.beta)
p                       = 1.0
q                       = 1.0
lrate                   = float(args.lrate)
epsilon                 = float(args.epsilon)
max_iter                = int(args.max_iter)
#kernel                  = str(args.kernel)
matrix_format           = str(args.matrix_format)
model_file		= str(args.model_file)
loss_file               = str(args.loss_file)
theta_plus              = float(args.theta_plus)
XI_plus                 = float(args.XI_plus)
theta_minus             = float(args.theta_minus)
XI_minus                = float(args.XI_minus)



if alpha+beta > 1.0:
	exit("alpha+beta should not be greater than 1.")


#by default, matrix is dense
is_sparse_calc 	= False
is_sparse_read 	= False

if matrix_format == None:
	matrix_format = "dense"

if ( matrix_format not in ["dense", "sparse"] ):
	sys.exit("matrix format show be either \"dense\" or \"sparse\".")
elif ( matrix_format == "sparse"):
	is_sparse_calc 	= True
	is_sparse_read 	= True




model_exists = os.path.exists(model_file)
if model_exists:
        exit("Warning: training not completed. Model file exist. ")



############################################
#read bioassays
############################################
kernel		= "linear"
all_bioassays 	= read_multi_bioassays(bioassay_id, bioassay_list_file, kernel, bioassay_comp_file, is_sparse_read)

dimension	= 0
if is_sparse_read:
	if kernel == "tanimoto":
		dimension 	= len(all_bioassays[bioassay_id][2])
		is_sparse_calc  = False
	elif kernel == "linear":
		dimension 	= all_bioassays[bioassay_id][2][0].shape[1]
else:
	if kernel == "tanimoto":
		dimension 	= len(all_bioassays[bioassay_id][2])
		is_sparse_calc  = False
	elif kernel == "linear":
                dimension 	= len(all_bioassays[bioassay_id][2][0])



###############################################
#gradient descent to find w
###############################################
w = loss_gradient_descent(all_bioassays, bioassay_id, dimension, alpha, L_c_option, \
				beta, p, q, lrate, epsilon, max_iter, model_file, is_sparse_calc, \
				loss_file, theta_plus, XI_plus, theta_minus, XI_minus)

###############################################
#write model
###############################################
write_model(w, model_file, is_sparse_calc)


