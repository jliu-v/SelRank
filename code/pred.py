from read import *
from functions import *
import sys
import os
import time
import argparse

#create argument parser
parser = argparse.ArgumentParser()

#add arguments
parser.add_argument("-dimension", help="Feature dimension")
parser.add_argument("-model_file", help="Path to output model file")
parser.add_argument("-test_feature_file", help="Path to testing feature file")
parser.add_argument("-pred_file", help="Path to output prediction file on testing set")
parser.add_argument("-matrix_format", help="Matrix format should be either \"dense\" or \"sparse\".")


args = parser.parse_args()


dimension               = int(args.dimension)
test_feature_file       = str(args.test_feature_file)
pred_file               = str(args.pred_file)
model_file		= str(args.model_file)
matrix_format		= str(args.matrix_format)




is_sparse_calc 	= False
is_sparse_read 	= False



#by default, set matrix format to dense
if matrix_format == None:
	matrix_format = "dense"

if ( matrix_format not in ["dense", "sparse"] ):
	sys.exit("Matrix format should be either \"dense\" or \"sparse\".")
elif ( matrix_format == "sparse"):
	is_sparse_calc 	= True
	is_sparse_read 	= True





################################
#read model
################################
#initialize model
w = None

#check if model file exists
model_exists = os.path.exists(model_file)

if model_exists:
        w = read_model(model_file, is_sparse_calc)
else:
	exit("Model file does not exist: "+model_file)


################################
#read test feature
################################
test = read_bioassay_feature(test_feature_file, dimension)
test_features_list   = test["features"]



################################
#prediction on test
################################
#predict
pred_list = predict(w, test_features_list, is_sparse_calc)

#write prediction, evaluations
write_prediction(pred_list, pred_file)




