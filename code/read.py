import sys
from functions import *
from scipy.sparse import csr_matrix



##########################################
#read single bioassay
##########################################
def read_bioassay_feature(feature_file, dimension, is_sparse):
	scores	 = []
	features = []
	with open(feature_file, "r") as read_feature:

                for line in read_feature:
                        parts = line.split()
			#read score
			scores.append(float(parts[0]))


                        #parse features
                        feature = [0.0 for i in range(dimension)]


                        for i in range(2, len(parts)):
                                token = parts[i].split(":")
                                idx = int(token[0])
                                val = float(token[1])
                                                
				feature[int(idx)-1] = val
			if(is_sparse):
				features.append(csr_matrix(feature))
			else:
				features.append(feature)
                        #features.append(csr_matrix(feature))


        read_feature.close()
	

	return {"scores": scores, "features": features}


##########################################
#read train bioassay
##########################################
def read_train_bioassay(relevance_indicator_file, feature_file, dimension, kernel, is_sparse):
	#read relevance indicators
	indicators 	= []
	x_indicators 	= []
	cids		= []
	with open(relevance_indicator_file, "r") as read_indicator:
		for line in read_indicator:
			parts = line.rstrip("\n").split()
			#append information
			indicators.append(parts[0])
			x_indicators.append(parts[1])
			cids.append(parts[2])
	read_indicator.close()


	#read features
	scores   = []
	features = []
	with open(feature_file, "r") as read_feature:
		for line in read_feature:
			parts = line.split()
			#read score
			scores.append(float(parts[0]))

			#parse features
			feature = [0.0 for i in range(dimension)]
			for i in range(2, len(parts)):
				token = parts[i].split(":")
				idx = token[0]
				val = token[1]

				feature[int(idx)-1] = float(val)
			if is_sparse:
				features.append(csr_matrix(feature))
			else:
				 features.append(feature)
	read_feature.close()


	#sort by scores from high to low
	sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

	sorted_indicators 	= [indicators[sorted_idx[i]] 	for i in range(len(sorted_idx))]
	sorted_scores     	= [scores[sorted_idx[i]]     	for i in range(len(sorted_idx))]
	sorted_features   	= [features[sorted_idx[i]]   	for i in range(len(sorted_idx))]
	sorted_x_indicators	= [x_indicators[sorted_idx[i]] 	for i in range(len(sorted_idx))]
	sorted_cids		= [cids[sorted_idx[i]] 		for i in range(len(sorted_idx))]
	if(kernel == "tanimoto"):
		kernel_matrix	  = tanimoto_kernel_matrix(sorted_features, sorted_features, is_sparse)
		return [sorted_indicators, sorted_scores, kernel_matrix, sorted_features, sorted_x_indicators, sorted_cids]

	elif(kernel == "linear"):
		return [sorted_indicators, sorted_scores, sorted_features, sorted_x_indicators, sorted_cids]



##########################################
#read bioassay list file
#assay list file should contain
#1. bioassay id
#2. path to training file
#3. path to training selectivity indicator file
#4. path to testing file
#5. path to testing indicator file
#6. baseline model file
#7. dimension
##########################################
def read_bioassay_list_file(bioassay_list_file):
	#read bioassay list as array
        id_list         	= []
	dimension_list		= []
        train_file_list 	= []
        train_ind_list  	= []
        baseline_model_list 	= []


	#read bioassay list file
        with open(bioassay_list_file, "r") as read_bioassay_list:
                for line in read_bioassay_list:
                        parts = line.strip('\t').split()
                        #parse each component
                        id_list.append(parts[0])
                        train_file_list.append(parts[1])
                        train_ind_list.append(parts[2])
			baseline_model_list.append(parts[3])
			dimension_list.append(int(parts[4]))
        read_bioassay_list.close()

	return id_list, train_file_list, train_ind_list, baseline_model_list, dimension_list



def read_comp_assay_file(assay_comp_file):

	comp_assay_appear = {}

	with open(assay_comp_file, "r") as reader:

		for line in reader:
			parts 	= line.strip("\n").split("\t")
			AID	= parts[0]
			CID 	= parts[1]
			if CID not in comp_assay_appear.keys():
				comp_assay_appear[CID] = []

			comp_assay_appear[CID].append(AID)
	reader.close()

	return comp_assay_appear





##########################################
#read bioassays from a list
##########################################
def read_multi_bioassays(bioassay_id, bioassay_list_file, kernel, assay_comp_file, is_sparse):
	"""
	kernel???
	"""
	#kernel = "linear"

	#read bioassay list 
	id_list, train_file_list, train_ind_list, baseline_model_list, dimension_list = read_bioassay_list_file(bioassay_list_file)
	
	#read each bioassay, and add to hashtable
	all_bioassays = {}

	###################################
	#read the target bioassay first
	###################################
	i = id_list.index(bioassay_id)
	#read training file
        #read sorted_indicators, sorted_scores, sorted_features, sorted_x_indicators, sorted_cids
        current_bioassay = read_train_bioassay(train_ind_list[i], train_file_list[i], dimension_list[i], 'linear', is_sparse)
        #read baseline model
	model_sparse = is_sparse
        if kernel == "tanimoto":
                model_sparse = False
        current_bioassay.append(read_model(baseline_model_list[i], model_sparse))
                

        #finish reading one bioassay
        all_bioassays[id_list[i]] = current_bioassay


	#####################################
	#read involved bioassays
	#####################################
	comp_assay 	= read_comp_assay_file(assay_comp_file)
	involved_assay 	= []
	CID_list 	= current_bioassay[4]
	x_selective_ind	= current_bioassay[3]
	for idx in range(len(CID_list)):
		if x_selective_ind[idx] == "1":
			involved_assay = involved_assay + comp_assay[CID_list[idx]]

	involved_assay = list(set(involved_assay))

	
	for AID in involved_assay:
		if AID == bioassay_id:
			continue

		#i is index
		i = id_list.index(AID)

		#read training file
		#read sorted_indicators, sorted_scores, sorted_features, sorted_x_indicators, sorted_cids
		current_bioassay = read_train_bioassay(train_ind_list[i], train_file_list[i], dimension_list[i], 'linear', is_sparse)
		#read baseline model
		model_sparse = is_sparse
		if kernel == "tanimoto":
			model_sparse = False
		current_bioassay.append(read_model(baseline_model_list[i], model_sparse))
		

		#finish reading one bioassay
		all_bioassays[id_list[i]] = current_bioassay
		
	"""
	all_bioassays:
	0.	train selective indicator
	1.	train score
	2.	train feature
        3.	train x_indicator
        4.	train cid
        5.	baseline model: w
	"""		
	return all_bioassays
		



#############################################
#read a test bioassay
#############################################
def read_test_bioassay(relevance_indicator_file, feature_file, dimension, kernel, is_sparse):
	#tempororily not using tanimoto kernal
	if kernel != "linear":
		exit("Tanimoto kernel temporarily not supported.")
	
        #read relevance indicators
        indicators 	= []
	x_indicators	= []
	cids		= []
        with open(relevance_indicator_file, "r") as read_indicator:
                for line in read_indicator:
                        parts = line.rstrip("\n").split()
                        #append information
                        indicators.append(parts[0])
                        x_indicators.append(parts[1])
                        cids.append(parts[2])

        read_indicator.close()


        #read features
        scores   = []
        features = []
        with open(feature_file, "r") as read_feature:
                for line in read_feature:
                        parts = line.split()
                        #read score
                        scores.append(float(parts[0]))

                        #parse features
                        feature = [0.0 for i in range(dimension)]
                        for i in range(2, len(parts)):
                                token = parts[i].split(":")
                                idx = token[0]
                                val = token[1]
	
                                feature[int(idx)-1] = float(val)
			if is_sparse:
	                        features.append(csr_matrix(feature))
			else:
				features.append(feature)
        read_feature.close()


        return [indicators, scores, features, x_indicators, cids]



#############################################
#read a model
#############################################
def read_model(model_file, is_sparse):
	w = []
	with open(model_file, "r") as reader:

		for line in reader:
    			pass
		
		parts = line.strip("\n").split(", ")
		for weight in parts:
			w.append(float(weight))


		del parts
	reader.close()

	if is_sparse:
		return csr_matrix(w)
	else:
		return w





#############################################
#read a prediction file
#############################################
def read_pred(pred_file):
        pred_list = []

        #open file
        with open(pred_file, "r") as reader:
                for line in reader:
                        pred_list.append(float(line.strip("\n")))
        reader.close()

        return pred_list







