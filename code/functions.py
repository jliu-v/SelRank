import pdb
import numpy as np
import copy
from scipy.sparse import csr_matrix
import sys

#########################################
#tanimoto function
#########################################
def tanimoto(a, b, is_sparse):

	
	if is_sparse:
		f_a  = float((a * a.transpose()).toarray())
		f_b  = float((b * b.transpose()).toarray())
        	f_ab = float((a * b.transpose()).toarray())
	
	else:
		f_a  = np.dot(a, a)
		f_b  = np.dot(b, b)
		f_ab = np.dot(a, b)


        return float(f_ab)/(f_a + f_b - f_ab)



#########################################
#generate tanimoto kernel matrix
#========================================
#all_bioassays is expected to be a python dictionary
#format = { bioassay_id:  [[relavant_indicators],[scores],[features] ] }
#bioassay = all_bioassays[bioassay_id]
#input is the a list of features from such bioassay
#the output is an M x N matrix 
#M = len(feature_list_A)
#N = len(feature_list_B)
#########################################
def tanimoto_kernel_matrix(train_feature_list, test_feature_list, is_sparse):
	#initialize
	all_features 	= train_feature_list + test_feature_list
	len_all 	= len(all_features)
	len_train 	= len(train_feature_list)
	
	#calculate
	t_kernel = [[tanimoto(all_features[i], train_feature_list[j], is_sparse) for j in range(len_train)] for i in range(len_all)]
	
	#return matrices
	train_kernel	= t_kernel[:len_train]
	test_kernel	= t_kernel[len_train:]
	return train_kernel, test_kernel 


#########################################
#score function
#########################################
def f(w, x, is_sparse):
	if is_sparse:
	        return float(w.dot(x.transpose()).toarray())
	else:
		return float(np.dot(w, x))

#########################################
#sigmoid function
#########################################
def sigmoid(x):
        return 1.0/(1+np.exp(-x))




#########################################
#delta(xi, xj) with weight vector w
#measure the score difference between xi
#and xj
#########################################
def delta(f_scores, xi, xj):
	#return f(w, xi, is_sparse)-f(w, xj, is_sparse)
	return (f_scores[xi] - f_scores[xj])



#########################################
#l(w, xi, xj)
#surrogate of indicator function
#########################################
def l(f_scores, xi, xj):
	return -np.log(sigmoid( delta(f_scores, xi, xj) ))





#########################################
#theta(xi, xj)
#derivative loss function on xi and xj
#xi and xj are indices
#########################################
def theta(f_scores, features, xi, xj, is_sparse):
	if is_sparse:
		return (  (1.0 - sigmoid(delta(f_scores, xi, xj)))   *   (-(features[xi] - features[xj]))  )
	else:
		return (  (1.0 - sigmoid(delta(f_scores, xi, xj)))   *   (-np.subtract(features[xi], features[xj]))  )



########################################
#ranking_percentile(f_scores, xi)
#get the ranking percentile of compound xi
#in the ranking list of f_scores
########################################
def ranking_percentile(f_scores, xi):
	xi_score 	= f_scores[xi]
	sorted_scores 	= sorted(list(set(f_scores)), key=float)

	return sorted_scores.index(xi_score)/float(len(sorted_scores))





#########################################
#nCI_surr(true_list, pred_list)
#surrogated non-concordance index of a ranking list
#f_scores are predicted scores based current w
#sorted by ground truth list descendingly
#########################################
def nCI_surr(f_scores, true_scores):
	#get the length of pred list
	#number of compounds
	size = len(f_scores)
	surr_loss = 0.0
	
	
	#iterate thru each pair of compounds
	#xi ranked higher than xj in ground truth
	for xi in range(0, size-1):
                        for xj in range(xi+1, size):
				
				#count only wrongly ranked pairs
				if f_scores[xi] > f_scores[xj] and true_scores[xi] > true_scores[xj]:
					continue
				elif f_scores[xi] == f_scores[xj] and true_scores[xi] == true_scores[xj]:
                                        continue
				elif f_scores[xi] < f_scores[xj] and true_scores[xi] < true_scores[xj]:
                                        continue

				#count only wrongly ranked pairs
                                surr_loss += l(f_scores, xi, xj)

	#return the normalized loss
	return surr_loss/(0.5*size*(size-1))





#########################################
#nCI_surr(true_list, pred_list)
#surrogated non-concordance index of a ranking list
#f_scores are predicted scores based current w
#sorted by ground truth list descendingly
#########################################
def nCI_surr_derivative(f_scores, true_scores, features, is_sparse, dimension):
        #get the length of pred list
        #number of compounds
        size = len(f_scores)
        dev  = None

	if is_sparse:
		dev = csr_matrix(  [0.0 for i in range(dimension)]  )
	else:
		dev = np.asarray([0.0 for i in range(dimension)])


        #iterate thru each pair of compounds
        #i ranked higher than j in ground truth
        for xi in range(0, size-1):
                        for xj  in range(xi+1, size):
				
				#count only wrongly ranked pairs
                                if f_scores[xi] > f_scores[xj] and true_scores[xi] > true_scores[xj]:
                                        continue
                                elif f_scores[xi] == f_scores[xj] and true_scores[xi] == true_scores[xj]:
                                        continue
                                elif f_scores[xi] < f_scores[xj] and true_scores[xi] < true_scores[xj]:
                                        continue

				#count only wrongly ranked pairs
				if is_sparse:
                                        dev = (dev + theta(f_scores, features, xi, xj, is_sparse))
                                else:
                                        dev = np.add(dev, theta(f_scores, features, xi, xj, is_sparse))

	#normalize
        if is_sparse:
        	dev = (dev / (0.5*size*(size-1)))
        else:
                dev = np.divide(dev,(0.5*size*(size-1)))

	

        #return the normalized dev
        return dev


#########################################
#loss of activity part
#sum of average loss over list(bioassays)
#========================================
#all_bioassays is expected to be a python dictionary
#format = { bioassay_id:  [[relavant_indicators],[scores],[features] ] }
#########################################
def L_c(all_bioassays, bioassay_id, w, L_c_option, is_sparse):
	#note: all features are expected to be sorted by true score from high low
        indicators	= all_bioassays[bioassay_id][0]
	t_scores	= all_bioassays[bioassay_id][1]
        features   	= all_bioassays[bioassay_id][2]

        surr_loss_sum = 0.0

	#computer score list
        f_scores = [f(w, x, is_sparse) for x in features]

	#L_c_option 1: L_b
	#use the nCI on the whole ranking list
	if L_c_option == "1":
		surr_loss_sum = nCI_surr(f_scores, t_scores)

	#L_c_option 2: L_s + L_a
	#sum of the nCI on selective and non-selective parts
	elif L_c_option == "2":
		#generate lists of relevant and irrelevant items
                relevant_f_scores   = []
		relevant_t_scores   = []
                irrelevant_f_scores = []
		irrelevant_t_scores = []
                for i in range(len(indicators)):
                        if(indicators[i] == "1"):
                                relevant_f_scores.append(f_scores[i])
				relevant_t_scores.append(t_scores[i])
                        elif(indicators[i] == "0"):
                                irrelevant_f_scores.append(f_scores[i])
				irrelevant_t_scores.append(t_scores[i])
                        else:
                                sys.exit("Relevance/Selectivity indicator should be either \"1\" or \"0\"")
		
		
		#sum up the nCI on selective and non-selective parts
		if len(relevant_f_scores) not in [0, 1]:
			surr_loss_sum += nCI_surr(relevant_f_scores, relevant_t_scores)
		if len(irrelevant_f_scores) not in [0, 1]:
			surr_loss_sum += nCI_surr(irrelevant_f_scores, irrelevant_t_scores)

	else:
		sys.exit("Invalid L_c_option. ")

	
        return surr_loss_sum




#########################################
#derivative of L_c
#########################################
def L_c_derivative(all_bioassays, bioassay_id, w, L_c_option, is_sparse, dimension):
	#note: all features are expected to be sorted by true score from high to low
        indicators = all_bioassays[bioassay_id][0]
	t_scores   = all_bioassays[bioassay_id][1]
        features   = all_bioassays[bioassay_id][2]


	#computer score list
        f_scores = [f(w, x, is_sparse) for x in features]


	dev0  = None

        if is_sparse:
                dev0 = csr_matrix(  [0.0 for i in range(dimension)]  )
        else:
                dev0 = np.asarray([0.0 for i in range(dimension)])


	#L_c_option 1: L_b
        #use the nCI on the whole ranking list
        if L_c_option == "1":
		loss_dev_sum = nCI_surr_derivative(f_scores, t_scores, features, is_sparse, dimension)


        #L_c_option 2: L_s + L_a
        #sum of the nCI on selective and non-selective parts
        elif L_c_option == "2":
                #generate lists of relevant and irrelevant items
                relevant_f_scores   = []
                irrelevant_f_scores = []
		relevant_features   = []
		irrelevant_features = []
                relevant_t_scores   = []
                irrelevant_t_scores = []

                for i in range(len(indicators)):
                        if(indicators[i] == "1"):
                                relevant_f_scores.append(f_scores[i])
				relevant_features.append(features[i])
				relevant_t_scores.append(t_scores[i])
                        elif(indicators[i] == "0"):
                                irrelevant_f_scores.append(f_scores[i])
				irrelevant_features.append(features[i])
				irrelevant_t_scores.append(t_scores[i])
                        else:
                                sys.exit("Relevance/Selectivity indicator should be either \"1\" or \"0\"")

                #sum up the nCI on selective and non-selective parts
		if len(relevant_f_scores) not in [0, 1]:
			relavant_loss_dev_sum	= nCI_surr_derivative(relevant_f_scores, relevant_t_scores, relevant_features, is_sparse, dimension)
			
		else:
			relavant_loss_dev_sum	= dev0
	
		if len(irrelevant_f_scores) not in [0, 1]:
			irrelevant_loss_dev_sum	= nCI_surr_derivative(irrelevant_f_scores, relevant_t_scores, irrelevant_features, is_sparse, dimension)
		else:
			irrelevant_loss_dev_sum = dev0
		
		loss_dev_sum = relavant_loss_dev_sum + irrelevant_loss_dev_sum		

	
	#invalid option
        else:
                sys.exit("Invalid L_c_option. ")


        return loss_dev_sum








########################################
#R_xi_plus(irrelevant_items, xi, f_scores, is_sparse)
#Reverse height of ONE selective compound
#R_xi = number of non-selective compound ranked above xi
########################################
def R_xi_plus(irrelevant_items, xi, f_scores, is_sparse):
        loss_sum = 0.0
        for xj in irrelevant_items:
                #if(f(w, xi, is_sparse) <= f(w, xj, is_sparse)):
                if( f_scores[xi] <= f_scores[xj] ):
                        loss_sum += l(f_scores, xi, xj)
        return loss_sum




########################################
#R_xi_plus_derivative(irrelevant_items, xi, f_scores, is_sparse)
#derivative of R_xi_plus function
########################################
def R_xi_plus_derivative(irrelevant_items, xi, f_scores, features, is_sparse, dimension):
	#initialize devirative
	dev  = None
        if is_sparse:
                dev = csr_matrix(  [0.0 for i in range(dimension)]  )
        else:
                dev = np.asarray([0.0 for i in range(dimension)])

	#iterate thru irrelevant items
        for xj in irrelevant_items:
                #add up the dev of surrogated indicator function
		#if the non-selective is ranked higher than selective
                if( f_scores[xi] <= f_scores[xj] ):
			if is_sparse:
                                dev = (dev + theta(f_scores, features, xi, xj, is_sparse))
                        else:
                                dev = np.add(dev, theta(f_scores, features, xi, xj, is_sparse))


        #return the dev
        return dev




########################################
#phi(ri_minus, ri_plus, XI)
#ranking percentile difference for
#push weight
########################################
def phi(ri_minus, ri_plus, XI):
	if ri_minus - ri_plus + XI > 0:
		return ri_minus - ri_plus + XI
	else:
		return 0



########################################
#g(xi)
#push-up weight for selective compound xi
########################################
def g(all_bioassays, bioassay_id, xi, baseline_scores, theta_plus, XI_plus, is_sparse):
        """
        all_bioassays:
        0.      train selective indicator
        1.      train score
        2.      train feature
        3.      train x_indicator
        4.      train cid
        5.      baseline model: w
        6.      test selective indicator
        7.      test scores
        8.      test features
        9.      test x_indicators
        10.     test cids
        """
	#get target assay selective compound information
	CID = all_bioassays[bioassay_id][4][xi]

	################################################
	#get ranking information of xi in target assay
	################################################
	#ranking percentile
	ri_plus			= ranking_percentile(baseline_scores, xi)
	cross_assay_count	= 0.0
	max_diff		= 0.0

	#find other x-selective compounds
	for AID in all_bioassays.keys():
		#skip the target bioassay
		if bioassay_id == AID:
			continue
		
		#read currect bioassay
		current_bioassay = all_bioassays[AID]
		
		#check if curreny assay contains the x-selective comp
		CIDs 				= current_bioassay[4]
		if CID not in CIDs:
			continue
		
		#get indicator files, check x_selective compound
		cross_xi = CIDs.index(CID)
		cross_relevance_indicator       = current_bioassay[0]
		cross_x_indicator		= current_bioassay[3]
		

		if 	cross_relevance_indicator[cross_xi] != "1" \
			and cross_x_indicator[cross_xi] != "1":
			exit("Something wrong with x_indicator. Assay:"+AID+" CID:"+CID)

		################################################
		#get the ranking percentile of x_selective compound
		################################################
		#compute the ranking scores
		cross_baseline_model	= current_bioassay[5]
		cross_features		= current_bioassay[2]
		if is_sparse:
			cross_f_scores  = [cross_features[i].dot(cross_baseline_model.transpose())[0][0] for i in range(len(cross_features))]
			#cross_f_scores  = cross_features.dot(cross_baseline_model)
		else:
			cross_f_scores	= np.dot(cross_features, cross_baseline_model)
	
		#get percentile
		ri_minus 		= ranking_percentile(cross_f_scores, cross_xi)
		
		if phi(ri_minus, ri_plus, XI_plus) > max_diff:
			max_diff = phi(ri_minus, ri_plus, XI_plus)
		"""
		print ri_plus
		print ri_minus
		print diff_sum
		exit()
		"""

        #target volumn means the difference of 1 and 
        #selective compounds' ranking percentile in target assay
        target_vol = 1.0 - ri_plus
	#cross volumn is the maximum ranking difference
        cross_vol  = max_diff

	#weight
	weight 	   = np.exp(theta_plus*(target_vol + cross_vol))

	return weight




########################################
#L_R_plus()
########################################
def L_R_plus(f_scores, relevant_items, irrelevant_items, p, is_sparse, push_up_weights):
	#if there is no relevant items, return 0
	if len(relevant_items) == 0:
		return 0

	L_R_plus_sum = 0.0
	#sum up the R_xi_plus for all selective compounds
	#each R_xi_plus is powered by p
	for xi in relevant_items:
		#print g(all_bioassays, bioassay_id, xi, f_scores, theta_plus, XI_plus)
		L_R_plus_sum += push_up_weights[xi] * \
				(R_xi_plus(irrelevant_items, xi, f_scores, is_sparse)) ** p

	#return the normalized sum of L_R_plus
	return L_R_plus_sum/(len(relevant_items))



########################################
#L_R_plus_derivative()
########################################
def L_R_plus_derivative(f_scores, relevant_items, irrelevant_items, features, p, is_sparse, dimension, push_up_weights):
	#initialize devirative
        dev  = None
        if is_sparse:
                dev = csr_matrix(  [0.0 for i in range(dimension)]  )
        else:
                dev = np.asarray([0.0 for i in range(dimension)])

	#if no relevant items, return 0
	if len(relevant_items) == 0:
                return dev

	#iterate thru selective compounds
	#sum up the dev of all selective compounds
	for xi in relevant_items:
		if is_sparse:
                        dev = 	dev + \
				push_up_weights[xi] * \
				p * \
				(R_xi_plus(irrelevant_items, xi, f_scores, is_sparse) ** (p-1) ) * \
				R_xi_plus_derivative(irrelevant_items, xi, f_scores, features, is_sparse, dimension)
                else:
			#print g(all_bioassays, bioassay_id, xi, f_scores, theta_plus, XI_plus)
			#print R_xi_plus(irrelevant_items, xi, f_scores, is_sparse) ** (p-1)
			#print R_xi_plus_derivative(irrelevant_items, xi, f_scores, features, is_sparse, dimension)
                        dev = 	np.add(dev, \
					push_up_weights[xi] * \
					p * \
					(R_xi_plus(irrelevant_items, xi, f_scores, is_sparse) ** (p-1) ) *\
					R_xi_plus_derivative(irrelevant_items, xi, f_scores, features, is_sparse, dimension)\
				)
	
	#normalize dev
	if is_sparse:
		dev = dev / len(relevant_items)
	else:
		dev = np.divide(dev, len(relevant_items))

	#return normalized dev
	return dev




########################################
#H_xj_minus(xj, f_scores, is_sparse)
#height of ONE x-selective compound
#(x-selective compound means this compound
#is selective in other bioassays but not in this one)
#H_xj = number of all compounds ranked below xj
########################################
def H_xj_minus(xj, f_scores, is_sparse):
        loss_sum = 0.0
	#iterate thru all compounds
        for xi in range(len(f_scores)):
		#if one compound is ranked below, count the loss
                if( f_scores[xi] <= f_scores[xj] and xi != xj):
                        loss_sum += l(f_scores, xi, xj)
        return loss_sum




########################################
#H_xj_minus_derivative(xj, f_scores, feature, is_sparse)
#derivative of H_xj_minus function
########################################
def H_xj_minus_derivative(xj, f_scores, features, is_sparse, dimension):
        #initialize devirative
        dev  = None
        if is_sparse:
                dev = csr_matrix(  [0.0 for i in range(dimension)]  )
        else:
                dev = np.asarray([0.0 for i in range(dimension)])

        #iterate thru irrelevant items
        for xi in range(len(f_scores)):
                #add up the dev of surrogated indicator function
                #if the non-selective is ranked higher than selective
                if( f_scores[xi] <= f_scores[xj] and xi != xj):
                        if is_sparse:
                                dev = (dev + theta(f_scores, features, xi, xj, is_sparse))
                        else:
                                dev = np.add(dev, theta(f_scores, features, xi, xj, is_sparse))


        #return the dev
        return dev

########################################
#h(xj)
#push-down weight function of
#x-selective compound xj
########################################
def h(all_bioassays, bioassay_id, xj, baseline_scores, theta_minus, XI_minus, is_sparse):
        """
        all_bioassays:
        0.      train selective indicator
        1.      train score
        2.      train feature
        3.      train x_indicator
        4.      train cid
        5.      baseline model: w
        6.      test selective indicator
        7.      test scores
        8.      test features
        9.      test x_indicators
        10.     test cids
        """
        #get target assay selective compound information
        CID = all_bioassays[bioassay_id][4][xj]

        ################################################
        #get ranking information of xj in target assay
        ################################################
        #ranking percentile
        rj_minus = ranking_percentile(baseline_scores, xj)
	rj_plus	 = 0.0
	assay_count = 0.0
        #find other x-selective compounds
        for AID in all_bioassays.keys():
                #skip the target bioassay
                if bioassay_id == AID:
                        continue

                #read currect bioassay
                current_bioassay = all_bioassays[AID]

                #check if curreny assay contains the x-selective comp
                CIDs                            = current_bioassay[4]
                if CID not in CIDs:
                        continue


                #get indicator files, check x_selective compound
                cross_xj = CIDs.index(CID)
                cross_relevance_indicator       = current_bioassay[0]
                cross_x_indicator               = current_bioassay[3]

                if cross_relevance_indicator[cross_xj] != "1":
			continue

                ################################################
                #get the ranking percentile of x_selective compound
                ################################################
                #compute the ranking scores
                cross_baseline_model    = current_bioassay[5]
                cross_features          = current_bioassay[2]
		if is_sparse:
			cross_f_scores  = cross_features.dot(cross_baseline_model)
		else:
	                cross_f_scores  = np.dot(cross_features, cross_baseline_model)
                #get percentile
                rj_plus = ranking_percentile(cross_f_scores, cross_xj)
		assay_count += 1

	if assay_count == 0:
		return 1.0
	elif assay_count > 1:
		exit("Wrong with getting CID: "+CID)


        #target volumn means the x-selective compounds' ranking percentile in target assay
        target_vol = rj_minus
        #cross volumn means the ranking diff
        cross_vol  = phi(rj_minus, rj_plus, XI_minus)
        #weight
        weight = np.exp(theta_minus*(target_vol + cross_vol))


        return weight





########################################
#L_H_minus()
#w: weight vector
#q: push-down power
#is_sparse: (bool) whether data are saved as sparse
########################################
def L_H_minus(all_bioassays, bioassay_id, w, q, is_sparse, push_down_weights, f_scores, x_selective_items):
	"""
        #note: all features are expected to be sorted by true score from high low
        indicators      = all_bioassays[bioassay_id][0]
        features        = all_bioassays[bioassay_id][2]
	x_indicators	= all_bioassays[bioassay_id][3]	
	"""

	"""
        #computer score list
        f_scores = [f(w, x, is_sparse) for x in features]
	"""
	
	"""
	#get the list of x-selective items
	x_selective_items = []
	for i in range(len(x_indicators)):
                if(x_indicators[i] == "1"):
                        x_selective_items.append(i)
                elif (x_indicators[i] != "0"):
                        sys.exit("X-selectivity indicator should be either \"1\" or \"0\"")
	"""
	if len(x_selective_items) == 0:
		return 0
	

        L_H_minus_sum = 0.0
        #sum up the L_H_minus for all x-selective compounds
        #each L_H_minus is powered by q
        for xj in x_selective_items:
                L_H_minus_sum += 	push_down_weights[xj] * \
					H_xj_minus(xj, f_scores, is_sparse) ** q

        #return the normalized sum of L_R_plus
        return L_H_minus_sum/len(x_selective_items)



########################################
#L_H_minus_derivative()
########################################
def L_H_minus_derivative(all_bioassays, bioassay_id, w, q, is_sparse, dimension, push_down_weights, f_scores, x_selective_items, features):
	"""
        #note: all features are expected to be sorted by true score from high low
        indicators      = all_bioassays[bioassay_id][0]
        features        = all_bioassays[bioassay_id][2]
	x_indicators    = all_bioassays[bioassay_id][3]
	"""

        #initialize devirative
        dev  = None
        if is_sparse:
                dev = csr_matrix(  [0.0 for i in range(dimension)]  )
        else:
                dev = np.asarray([0.0 for i in range(dimension)])

	"""
        #computer score list
        f_scores = [f(w, x, is_sparse) for x in features]
        
	#generate lists of x-selective items
        x_selective_items = []
        for i in range(len(x_indicators)):
                if(x_indicators[i] == "1"):
                        x_selective_items.append(i)
                elif (x_indicators[i] != "0"):
                        sys.exit("X-selectivity indicator should be either \"1\" or \"0\"")
	"""
	if len(x_selective_items) == 0:
                return dev

        #iterate thru selective compounds
        #sum up the dev of all selective compounds
        for xj in x_selective_items:
                if is_sparse:
                        dev =   dev + \
                                push_down_weights[xj] * \
				q * \
                                (H_xj_minus(xj, f_scores, is_sparse) ** (q-1) ) * \
                                H_xj_minus_derivative(xj, f_scores, features, is_sparse, dimension)
                else:
                        dev =   np.add(dev, \
					push_down_weights[xj] * \
					q * \
                                        (H_xj_minus(xj, f_scores, is_sparse) ** (q-1) ) *\
                                	H_xj_minus_derivative(xj, f_scores, features, is_sparse, dimension)\
                                )

        #normalize dev
        if is_sparse:
                dev = dev / len(x_selective_items)
        else:
                dev = np.divide(dev, len(x_selective_items))

	#return the normalized dev
	return dev






#########################################
#L()
#overall loss
#L = alpha * L_c + beta * L_R_plus + (1 - alpha - beta)L_H_minus
#########################################
def L(all_bioassays, bioassay_id, w, alpha, L_c_option, beta, p, q, is_sparse, push_up_weights, push_down_weights, f_scores, relevant_items, irrelevant_items, x_selective_items):
	#Loss is the weighted sum of three parts
	L_sum = 0.0

	if (1.0-alpha-beta) != 0.0:
		L_sum += (1.0-alpha-beta) * L_c(all_bioassays, bioassay_id, w, L_c_option, is_sparse)

	if alpha != 0.0:
		L_sum += alpha  * L_R_plus(f_scores, relevant_items, irrelevant_items, p, is_sparse, push_up_weights)

	if beta != 0.0:
		L_sum += beta * L_H_minus(all_bioassays, bioassay_id, w, q, is_sparse, push_down_weights, f_scores, x_selective_items)

	#return the weighted sum
	return L_sum


#########################################
#L_derivative()
#patial derivative of L on w
#########################################
def L_derivative(all_bioassays, bioassay_id, w, alpha, L_c_option, beta, p, q, is_sparse, dimension, push_up_weights, push_down_weights, f_scores, relevant_items, irrelevant_items, x_selective_items, features):

	#initialize devirative
        dev  = None
        if is_sparse:
                dev = csr_matrix(  [0.0 for i in range(dimension)]  )
        else:
                dev = np.asarray([0.0 for i in range(dimension)])


	#the derivative is the weighted sum of three parts
	if (1.0-alpha-beta) != 0.0:
                dev += (1.0-alpha-beta) * L_c_derivative(all_bioassays, bioassay_id, w, L_c_option, is_sparse, dimension)
	
	if alpha != 0.0:
		dev += alpha  * L_R_plus_derivative(f_scores, relevant_items, irrelevant_items, features, p, is_sparse, dimension, push_up_weights)

	if beta != 0.0:
                dev += 	beta * L_H_minus_derivative(all_bioassays, bioassay_id, w, q, is_sparse, dimension, push_down_weights, f_scores, x_selective_items, features)
	

	return dev



#############################################
#write model
#############################################
def write_model(w, model_file, is_sparse):
        with open(model_file, "w") as writer:
                the_w = None
                if is_sparse:
                        the_w = w.toarray()[0]
                else:
                        the_w = w
                row_str = ", ".join(["%.6g"%(the_w[i]) for i in range(len(the_w))])
                writer.write(row_str+"\n")

        writer.close()




#############################################
#write loss history
#############################################
def write_loss(l_history, loss_file):
        with open(loss_file, "w") as writer:
                for loss in l_history:
                        writer.write("%.6g\n"%(loss))
        writer.close()




#########################################
#gradient descent
#all_bioassays is all bioassay information to be used
#dimension is the (max) dimension of features
#p is the power defined by user for power-push
#lrate is th learning rate for gradient descent
#alpha is the weight of selectivity push-up
#beta is the weight of x-selectivity push-down
#epsilon is the threshold of gradient descent convergence
#max_iter is the max number of iterations for gradient descent
#########################################
def loss_gradient_descent(all_bioassays, bioassay_id, dimension, alpha, L_c_option, beta, p, q, lrate, epsilon, max_iter, model_file, is_sparse, loss_file, theta_plus, XI_plus, theta_minus, XI_minus):
	#np.random.seed(0)
	#initialize w as random
	#w = np.random.rand(1, dimension)
	#^^^^#
	w = all_bioassays[bioassay_id][5]
	if is_sparse:
		w = csr_matrix(w)
	else:
		w = np.asarray(w)


	##############################
        #get information of the target bioassay
        ##############################
	indicators      = all_bioassays[bioassay_id][0]
	t_scores        = all_bioassays[bioassay_id][1]
        features        = all_bioassays[bioassay_id][2]
        x_indicators    = all_bioassays[bioassay_id][3]
        baseline_w      = all_bioassays[bioassay_id][5]



	##############################
	#calculate baseline score
	##############################
	f_scores = [f(baseline_w, x, is_sparse) for x in features]
	

	##############################
        #get the list of selective 
	#and non-selective
        ##############################
	#generate lists of selective, non-selective   
        relevant_items    = []
        irrelevant_items  = []
        for i in range(len(indicators)):
                if(indicators[i] == "1"):
                        relevant_items.append(i)
                elif(indicators[i] == "0"):
                        irrelevant_items.append(i)
                else:
                        sys.exit("Relevance/Selectivity indicator should be either \"1\" or \"0\"")

	#generate lists of x-selective items
        x_selective_items = []
        for i in range(len(x_indicators)):
                if(indicators[i] == "0" and x_indicators[i] == "1"):
                        x_selective_items.append(i)
                elif (x_indicators[i] != "0" and x_indicators[i] != "1"):
                        sys.exit("X-selectivity indicator should be either \"1\" or \"0\"")


	

	##############################
        #calculate push weights (list)
        ##############################
	push_up_weights		= [0.0 for i in range(len(indicators))]
	push_down_weights	= [0.0 for i in range(len(x_indicators))]
	for xi in relevant_items:
		push_up_weights[xi] 	= g(all_bioassays, bioassay_id, xi, f_scores, theta_plus, XI_plus, is_sparse)
	for xj in x_selective_items:
		push_down_weights[xj]	= h(all_bioassays, bioassay_id, xj, f_scores, theta_minus, XI_minus, is_sparse)

	#initial loss
	#l1 = alpha * activity_loss(all_bioassays, w, is_sparse) + (1.0 - alpha) * selectivity_loss(all_bioassays, w, p, is_sparse)
	l1  = L(all_bioassays, bioassay_id, w, alpha, L_c_option, beta, p, q, is_sparse, push_up_weights, push_down_weights, f_scores, relevant_items, irrelevant_items, x_selective_items)
	#print l1

	#w_history = []
	l_history = [l1]
	#iteration
	for i in range(max_iter):

		#copy the loss of last iteration as l0
		l0 = copy.deepcopy(l1)
		
		#find the derivative
		loss_dev = L_derivative(all_bioassays, bioassay_id, w, alpha, L_c_option, beta, p, q, is_sparse, dimension, push_up_weights, push_down_weights, f_scores, relevant_items, irrelevant_items, x_selective_items, features)
		#step to the gradient with a length of lrate
		w = (w - (lrate * loss_dev))

		#computer score list
	        f_scores = [f(w, x, is_sparse) for x in features]

		#get the loss with new w
		l1  = L(all_bioassays, bioassay_id, w, alpha, L_c_option, beta, p, q, is_sparse, push_up_weights, push_down_weights, f_scores, relevant_items, irrelevant_items, x_selective_items)

			
		#check if stop condition is met
		if(l1 == 0.0):
			break
		if( abs(l0-l1)/abs(l0) <= epsilon):
			break


	if(loss_file != None):
		write_loss(l_history, loss_file)

	return w




#########################################
#predict a list of features
#########################################
def predict(w, feature_list, is_sparse):
	pred_list = []
	for x in feature_list:
		pred_list.append(f(w, x, is_sparse))

	return pred_list
		


#########################################
#write a list of predictions to a file
#########################################
def write_prediction(pred_list, pred_file):
	with open(pred_file, "w") as write_pred:
		for prediction in pred_list:
			write_pred.write("%.6g\n" % prediction)

	write_pred.close()



#########################################
#CI on prediction
#########################################
def pred_CI(pred_list, true_list):
	correct_pair_count = 0.0
	if(len(pred_list) != len(true_list)):
		sys.exit("Prediction and True score list size do not match.")

	size = len(pred_list)
	for i in range(size-1):
		for j in range(i+1, size):
			if pred_list[i] > pred_list[j] and true_list[i] > true_list[j]:
				correct_pair_count += 1
			elif pred_list[i] < pred_list[j] and true_list[i] < true_list[j]:
				correct_pair_count += 1
			elif pred_list[i] == pred_list[j] and true_list[i] == true_list[j]:
                                correct_pair_count += 1
	return correct_pair_count/(0.5*size*(size-1))




##############################################################
#sCI: CI among selective part
##############################################################
def pred_sCI(pred_list, true_list, relevance_indicator_list):
        if(len(pred_list) != len(true_list)):
                sys.exit("Prediction and True score list size do not match.")



        active_idx      = []
        selective_idx   = []


        size = len(relevance_indicator_list)
        for i in range(size):
                if relevance_indicator_list[i] == "1":
                        selective_idx.append(i)
                elif relevance_indicator_list[i] == "0":
                        active_idx.append(i)
                else:
                        sys.exit("Selective indicator should be either 1 or 0.")

        CI_S = 0.0

        #CI among selective compounds
        size_s = len(selective_idx)
        if size_s not in [0, 1]:

                correct_pair_count_s = 0.0
                for s_i in range(size_s-1):
                        for s_j in range(s_i+1, size_s):
                                i = selective_idx[s_i]
                                j = selective_idx[s_j]
                                if pred_list[i] > pred_list[j] and true_list[i] > true_list[j]:
                                        correct_pair_count_s += 1
                                elif pred_list[i] < pred_list[j] and true_list[i] < true_list[j]:
                                        correct_pair_count_s += 1
                                elif pred_list[i] == pred_list[j] and true_list[i] == true_list[j]:
                                        correct_pair_count_s += 1
                CI_S = correct_pair_count_s/(0.5*size_s*(size_s-1))

        return CI_S




##############################################################
#aCI: CI among active part
##############################################################
def pred_aCI(pred_list, true_list, relevance_indicator_list):
	if(len(pred_list) != len(true_list)):
                sys.exit("Prediction and True score list size do not match.")



        active_idx      = []
        selective_idx   = []


        size = len(relevance_indicator_list)
        for i in range(size):
                if relevance_indicator_list[i] == "1":
                        selective_idx.append(i)
                elif relevance_indicator_list[i] == "0":
                        active_idx.append(i)
                else:
                        sys.exit("Selective indicator should be either 1 or 0.")

	CI_A = 0.0

	#CI among active compounds
        size_a = len(active_idx)
        if size_a not in [0, 1]:
                correct_pair_count_a = 0.0
                for a_i in range(size_a-1):
                        for a_j in range(a_i+1, size_a):
                                i = active_idx[a_i]
                                j = active_idx[a_j]
                                if pred_list[i] > pred_list[j] and true_list[i] > true_list[j]:
                                        correct_pair_count_a += 1
                                elif pred_list[i] < pred_list[j] and true_list[i] < true_list[j]:
                                        correct_pair_count_a += 1
                                elif pred_list[i] == pred_list[j] and true_list[i] == true_list[j]:
                                        correct_pair_count_a += 1
                CI_A = correct_pair_count_a/(0.5*size_a*(size_a-1))

	return CI_A





#########################################
#SCI on prediction
#SCI = CI_A + CI_S
#########################################
def pred_SCI(pred_list, true_list, relevance_indicator_list):
	return pred_aCI(pred_list, true_list, relevance_indicator_list) + pred_sCI(pred_list, true_list, relevance_indicator_list)


#########################################
#SI on prediction
#SI = (sum(selective ranking position)/selective_count)/number_comps
#########################################
def pred_SI(pred_list, relevance_indicator_list):

	if(len(pred_list) != len(relevance_indicator_list)):
                sys.exit("Prediction and selective indication list size do not match.")
	size = len(pred_list)

	sorted_uniq_score = sorted(list(set(pred_list)), reverse=True)

	position = 0.0
        count    = 0.0
	for i in range(size):
                if(relevance_indicator_list[i] == "1"):
			position += float(sorted_uniq_score.index(pred_list[i]))+1.0
                        count    += 1.0

	#if there is no selective compound, return 0
        if count == 0.0:
                return 0.0

	return (position/count)/float(size)




#########################################
#rxSI on prediction
#########################################
def pred_rxSI(pred_list, x_indicator_list):

        if(len(pred_list) != len(x_indicator_list)):
                sys.exit("Prediction and x-selective indication list size do not match.")
        size = len(pred_list)

        sorted_uniq_score = sorted(list(set(pred_list)), reverse=False)

        position = 0.0
        count    = 0.0
        for i in range(size):
                if(x_indicator_list[i] == "1"):
                        position += float(sorted_uniq_score.index(pred_list[i]))+1.0
                        count    += 1.0

        #if there is no selective compound, return 0
        if count == 0.0:
                return 0.0

        return (position/count)/float(size)









#########################################
#PI on prediction
#PI = sum(1/selective_ranking_position)/num_selective
#########################################
def pred_PI(pred_list, relevance_indicator_list):

	if(len(pred_list) != len(relevance_indicator_list)):
                sys.exit("Prediction and selective indication list size do not match.")

        sorted_uniq_score = sorted(list(set(pred_list)), reverse=True)

        position = 0.0
        count    = 0.0
        for i in range(len(relevance_indicator_list)):
                if(relevance_indicator_list[i] == "1"):
                        position += 1.0/(float(sorted_uniq_score.index(pred_list[i]))+1.0)
                        count    += 1.0

	#if there is no selective compound, return 0
	if count == 0.0:
		return 0.0

        return position/count



#########################################
#rxPI on prediction
#rxPI = sum(1/x_selective_ranking_position)/num_xselective
#########################################
def pred_rxPI(pred_list, x_indicator_list):

        if(len(pred_list) != len(x_indicator_list)):
                sys.exit("Prediction and x-selective indication list size do not match.")

        sorted_uniq_score = sorted(list(set(pred_list)), reverse=False)

        position = 0.0
        count    = 0.0
        for i in range(len(x_indicator_list)):
                if(x_indicator_list[i] == "1"):
                        position += 1.0/(float(sorted_uniq_score.index(pred_list[i]))+1.0)
                        count    += 1.0

        #if there is no selective compound, return 0
        if count == 0.0:
                return 0.0

        return position/count





#########################################
#WPI on prediction: weighted PI
#WPI = sum(1/selective_ranking_position) / sum(1/all_ranking_position)
#########################################
def pred_WPI(pred_list, relevance_indicator_list):

        if(len(pred_list) != len(relevance_indicator_list)):
                sys.exit("Prediction and selective indication list size do not match.")

        sorted_uniq_score = sorted(list(set(pred_list)), reverse=True)

        position_s	= 0.0
	position_a	= 0.0
        count    	= 0.0
        for i in range(len(relevance_indicator_list)):
		position_a += 1.0/(float(sorted_uniq_score.index(pred_list[i]))+1.0)
                if(relevance_indicator_list[i] == "1"):
                        position_s += 1.0/(float(sorted_uniq_score.index(pred_list[i]))+1.0)
                        count    += 1.0

	#if there is no selective compound, return 0
        if count == 0.0:
                return 0.0

        return position_s/position_a



#########################################
#rxWPI on prediction: weighted PI
#rxWPI = sum(1/x_selective_ranking_position) / sum(1/all_ranking_position)
#########################################
def pred_rxWPI(pred_list, x_indicator_list):

        if(len(pred_list) != len(x_indicator_list)):
                sys.exit("Prediction and x-selective indication list size do not match.")

        sorted_uniq_score = sorted(list(set(pred_list)), reverse=False)

        position_s      = 0.0
        position_a      = 0.0
        count           = 0.0
        for i in range(len(x_indicator_list)):
                position_a += 1.0/(float(sorted_uniq_score.index(pred_list[i]))+1.0)
                if(x_indicator_list[i] == "1"):
                        position_s += 1.0/(float(sorted_uniq_score.index(pred_list[i]))+1.0)
                        count    += 1.0

        #if there is no selective compound, return 0
        if count == 0.0:
                return 0.0

        return position_s/position_a


#########################################
#precision at k
#precision(k)
#########################################
def precision_at_k(sorted_relevance_indicator_list, k):
	if(k < 1):
		sys.exit("Precision@k. k must be positive integer.")

	count = 0.0

	if sorted_relevance_indicator_list[k-1] == "0":
		return count

	for i in range(k):
		if (sorted_relevance_indicator_list[i] == "1"):
			count += 1

	return count/k


#########################################
#ap@n
#ap_at_n(pred_list, indicator_list, n)
#########################################
def ap_at_n(pred_list, relevance_indicator_list, n):
	if(len(pred_list) != len(relevance_indicator_list)):
                sys.exit("Prediction and selective indication list size do not match.")
        if(n < 1):
                sys.exit("AP@n. n must be positive integer.")	
	
	#m is the number of relevant
	m = 0
	for i in range(len(relevance_indicator_list)):
		if(relevance_indicator_list[i] == "1"):
			m += 1

	if m == 0:
		return 0.0

	array           = [[pred_list[i], relevance_indicator_list[i]] for i in range(len(pred_list))]
        sorted_array    = sorted(array, key=lambda x: x[0], reverse=True)

	sorted_pred_list		= [sorted_array[i][0] for i in range(len(pred_list))]
	sorted_relevance_indicator_list = [sorted_array[i][1] for i in range(len(pred_list))]

	p_sum = 0.0
	n = min(n, len(pred_list))
	for k in range(n):
		p_sum += precision_at_k(sorted_relevance_indicator_list, k+1)


	return p_sum/min(m, n)



#########################################
#write one value
#########################################
def write_value(value, file_name):
	with open(file_name, "w") as write_val:
		write_val.write("%.6g\n" % value)
	write_val.close()


#########################################
#write value array
#########################################
def write_value_array(value_array, file_name):
	val_str = "\t".join(["%.6f"%(value_array[i]) for i in range(len(value_array))])
	

        with open(file_name, "w") as write_val:
                write_val.write("%s\n" % val_str)
        write_val.close()



















