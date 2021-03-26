import sys, string, re, random
import numpy as np

from random import randrange
from collections import defaultdict, OrderedDict
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

#Varvara Stein

#==========HOW TO RUN========
#pa3.py pa3_input_text.txt pa3_B.txt pa3_T.txt



def sigmoid(w, x):
	return 1/(1 + np.exp(-w*x, dtype=np.float32))


def preprocess_text(filename):
	cleaned = []
	reg = r'”|“'
	with open(input, 'r', encoding='utf-8') as filename:
		for lines in filename:
			temp_txt = lines.lower()
			temp_txt = temp_txt.translate(str.maketrans('', '', string.punctuation))
			temp_txt = temp_txt.split(' ')
			temp_txt = list(map(lambda s: s.strip(), temp_txt))
			for item in temp_txt:
				new = re.sub(reg, '', item)
				cleaned.append(new)
					
	return cleaned


def preprocess_b(b):
	with open(b, 'r', encoding = 'utf-8') as bas:
		basis = [word for line in bas for word in line.split('\n')]
		basis = [i for i in basis if i]
	
	return 	basis
	

def preprocess_t(t):
	"""
	Preprocess targets and labels
	:return: list of tuples [('war', 0), ('fight', 0),...]
	"""
	WAR = 0
	PEACE = 1
	target_list = []
	toks = []
	labels = []
	with open(t, 'r', encoding = 'utf-8') as targ:
		target = [word for line in targ for word in line.split('\t')]
		target = [i for i in target if i]
		new_list = [target[i:i+2] for i in range(0, len(target), 2)]
		for elems in new_list:
			toks.append(elems[0])
			if elems[1] == 'WAR\n':
				labels.append(WAR)
			else:
				labels.append(PEACE)
	target_list = list(zip(toks, labels))
	
	return target_list


def get_co_occurrence(words, bas, targ):
	"""
	Get sparse vector matrix
	:return: dictionary with target word as key and counts of target and basis cooccurrences as value
	"""
	data = defaultdict(list)
	for t in targ:
		data[t[0]] = [0 for elem in bas]
	
	for i in range(len(words)):
		if words[i] in data.keys():
			idx = -1
			if(i>=2):
				if( words[i-2] in bas):
					idx = bas.index(words[i-2])
			if(i>=1):
				if( words[i-1] in bas):
					idx = bas.index(words[i-1])
			if(i<len(words)-1):    
				if( words[i+1] in bas):
					idx = bas.index(words[i+1])
			if(i<len(words)-2):
				if( words[i+2] in bas):
					idx = bas.index(words[i+2])
			if(i==0):
				if(words[i+1] in bas):
					idx = bas.index(words[i+1])
				if( words[i+2] in bas):
					idx = bas.index(words[i+2])	
	  
			if(idx != -1):
				data[words[i]][idx]+=1
	
	return data


def initialize_weights(basis):
	w =[]
	for i in range(len(basis)):
		w.append(0)
		
	return w	


def sparse_cross_validation_split(sparse_matrix, n_folds):
	"""
	Split sparse vector dictionary into k folds and add labels
	:return: shuffled nested list with vector and label [(array([0, 8, ...]), 0)]
	"""
	value_list = []
	labels = []
	sparse_split = []
	shuffled_list = []
	for k, v in sparse_matrix.items():
		value_list.append(v)
	
	for i in range(len(value_list)):
		if i <=20:
			labels.append(0)
		else:
			labels.append(1)
			
	shuffled_list = list(zip(value_list, labels))		
	random.shuffle(shuffled_list)
	random.seed(4)
	fold_size = int(len(value_list) / n_folds)
	sparse_split = [shuffled_list[i:i + fold_size] for i in range(0, len(shuffled_list), fold_size)]
	
	return sparse_split


def get_dense_vec(input, target):
	"""
	Get dense vector from gensim model
	:return: The dictionary {'russia': (array([-0.4827912 ,...]), 0)}
	"""
	text = []
	vector_dict = defaultdict(list)
	with open(input, 'r', encoding='utf-8') as filename:
		for line in filename:
			line_low = line.lower()
			sents = sent_tokenize(line_low)
			for sent in sents:
				tmp = word_tokenize(sent)
				text.append(tmp)
	model = Word2Vec(text, min_count=1, size=83, workers=1, window=2, iter=60)
	for tpl in target:
		for token in tpl:
			if token in model:
				vector_dict[token] = (model[token], tpl[1])
	#print(vector_dict)
	return (vector_dict) 


def cross_validation_split(vector_dict, n_folds):
	"""
	Split dense vector dictionary into k folds
	:return: shuffled nested list with vector and label [(array([-0.4827912 ,...]), 0)]
	"""
	value_list = []
	values_split = []
	items = list(vector_dict.items())
	random.shuffle(items)
	shuffled_dict = OrderedDict(items)
	for k, v in shuffled_dict.items():
		value_list.append(v)
		
	fold_size = int(len(value_list) / n_folds)
	values_split = [value_list[i:i + fold_size] for i in range(0, len(value_list), fold_size)]
	random.seed(4)
	
	return values_split


def train(training_data, weights):
	eta = 0.2 #to control the learning rate
	fin = []
	for iter in range(100):
		for fold in training_data:
			for vecs, label in fold:
				res = sigmoid(2,np.dot(vecs, weights) - 0.5)
				error = label - res
				if abs(error) > 0.0:
					for i, val in enumerate(vecs):
						weights[i]+= val * error * eta
					
	return weights			


def get_single_predicted(test_set, calculated_w):
	single_predicted = []
	for fold in test_set: #[[vec, label],[],[]] -> one fold
		for elem in fold: #[vec, label]
			labels = sigmoid(2,np.dot(elem[0], calculated_w) - 0.5)
			single_predicted.append(labels)
		
	return single_predicted		


def get_predicted(test_set, calculated_w):
	predicted = []
	for elem in test_set: #[[vec, label]] -> one fold
		labels = sigmoid(2,np.dot(elem[0], calculated_w) - 0.5)
		predicted.append(labels)
			
	return predicted		
	

def get_single_accuracy(actual_list, predicted):
	correct = 0
	correct_labels = []
	sum_list = []
	for item in actual_list:
		for lab in item:
			correct_labels.append(lab[1])
	labs_predicted_list = list(zip(correct_labels, predicted))
	for item in labs_predicted_list:
		if (item[0] == 0 and item[1] < 0.5) or (item[0] == 1 and item[1] > 0.5):
			correct+=1
	print(labs_predicted_list)		
	print('correct ', correct)
	# for (item1, item2) in zip(correct_labels, predicted):
		# sum_list.append(item1-item2)
	# print('sum_list ', sum_list)
	# for nums in sum_list:
		# if nums < 0.3 or 0 >= nums > -0.5:
			# correct += 1
#if (correct_result == 0 and predicted_result < 0.5) or (correct_result == 1 and predicted_result > 0.5):
		
	return ((correct/ len(correct_labels))*100.0)

	
def get_accuracy(actual_list, predicted):
	correct = 0
	sum_list = []
	correct_labels = []
	for item in actual_list:
		correct_labels.append(item[1])
	labs_predicted_list = list(zip(correct_labels, predicted))
	for item in labs_predicted_list:
		if (item[0] == 0 and item[1] < 0.5) or (item[0] == 1 and item[1] > 0.5):
			correct+=1
	#print(correct)
	# for (item1, item2) in zip(correct_labels, predicted):
		# sum_list.append(item1-item2)
	# for nums in sum_list:
		# if nums < 0.3 or 0 >= nums > -0.5:
			# correct += 1
			
	return (correct/ float(len(correct_labels))*100.0)

	
if __name__ == '__main__':
	input = sys.argv[1]
	t = sys.argv[3]
	b = sys.argv[2]
	
	text = preprocess_text(input)
	bas = preprocess_b(b)
	targ = preprocess_t(t)
	matrix = get_co_occurrence(text, bas, targ)
	sparse_folds = sparse_cross_validation_split(matrix, n_folds=5)
	
	weights_nulls = initialize_weights(bas)
	
	dense = get_dense_vec(input, targ)
	folds = cross_validation_split(dense, n_folds=5)
	
	dense_cross_no_1 = [folds[0], folds[2], folds[3], folds[4]]
	dense_cross_no_2 = [folds[0], folds[1], folds[3], folds[4]]
	dense_cross_no_3 = [folds[0], folds[1], folds[2], folds[4]]
	
	dense_single_weights = train(folds, weights_nulls)
	dense_weights_no_4 = train(folds[0:4], weights_nulls) #test fold 5
	dense_weights_no_0 = train(folds[1:5], weights_nulls) #test fold 1
	dense_weights_no_1 = train(dense_cross_no_1, weights_nulls) #test fold 2
	dense_weights_no_2 = train(dense_cross_no_2, weights_nulls) #test fold 3
	dense_weights_no_3 = train(dense_cross_no_3, weights_nulls) #test fold 4
	
	dense_predicted_single = get_single_predicted(folds, dense_single_weights)
	#print(dense_predicted_single)
	dense_predicted_1 = get_predicted(folds[4], dense_weights_no_4)
	dense_predicted_2 = get_predicted(folds[0], dense_weights_no_0)
	dense_predicted_3 = get_predicted(folds[1], dense_weights_no_1)
	dense_predicted_4 = get_predicted(folds[2], dense_weights_no_2)
	dense_predicted_5 = get_predicted(folds[3], dense_weights_no_3)
	
	dense_single = get_single_accuracy(folds, dense_predicted_single)
	#print(dense_single)
	dense_eval_1 = get_accuracy(folds[4], dense_predicted_1)
	dense_eval_2 = get_accuracy(folds[0], dense_predicted_2)
	dense_eval_3 = get_accuracy(folds[1], dense_predicted_3)
	dense_eval_4 = get_accuracy(folds[2], dense_predicted_4)
	dense_eval_5 = get_accuracy(folds[3], dense_predicted_5)
	
	sparse_cross_no_1 = [sparse_folds[0], sparse_folds[2], sparse_folds[3], sparse_folds[4]]
	sparse_cross_no_2 = [sparse_folds[0], sparse_folds[1], sparse_folds[3], sparse_folds[4]]
	sparse_cross_no_3 = [sparse_folds[0], sparse_folds[1], sparse_folds[2], sparse_folds[4]]
	
	sparse_single_weights = train(sparse_folds, weights_nulls)
	sparse_weights_no_4 = train(sparse_folds[0:4], weights_nulls) #test fold 5 
	sparse_weights_no_0 = train((sparse_folds[1:5]), weights_nulls) #test fold 1 
	sparse_weights_no_1 = train(sparse_cross_no_1, weights_nulls) #test fold 2 
	sparse_weights_no_2 = train(sparse_cross_no_2, weights_nulls) #test fold 3 
	sparse_weights_no_3 = train(sparse_cross_no_3, weights_nulls) #test fold 4 
	
	sparse_predicted_single = get_single_predicted(sparse_folds, sparse_single_weights)
	sparse_predicted_1 = get_predicted(sparse_folds[4], sparse_weights_no_4)
	sparse_predicted_2 = get_predicted(sparse_folds[0], sparse_weights_no_0)
	sparse_predicted_3 = get_predicted(sparse_folds[1], sparse_weights_no_1)
	sparse_predicted_4 = get_predicted(sparse_folds[2], sparse_weights_no_2)
	sparse_predicted_5 = get_predicted(sparse_folds[3], sparse_weights_no_3)
	
	sparse_single = get_single_accuracy(sparse_folds, sparse_predicted_single)
	#print(sparse_single)
	sparse_eval_1 = get_accuracy(sparse_folds[4], sparse_predicted_1)
	sparse_eval_2 = get_accuracy(sparse_folds[0], sparse_predicted_2)
	sparse_eval_3 = get_accuracy(sparse_folds[1], sparse_predicted_3)
	sparse_eval_4 = get_accuracy(sparse_folds[2], sparse_predicted_4)
	sparse_eval_5 = get_accuracy(sparse_folds[3], sparse_predicted_5)
	
	sparse_AVG = (sparse_eval_1+sparse_eval_2+sparse_eval_3+sparse_eval_4+sparse_eval_5)/len(sparse_folds)
	dense_AVG = (dense_eval_1+dense_eval_2+dense_eval_3+dense_eval_4+dense_eval_5)/len(folds)
	
	print(str('Results').center(30))
	print("{:<14}{:<11}{:<11}".format('evaluation','sparse', 'dense'))
	print("{:<14}{:<11}{:<11}".format('single', round(sparse_single, 2), round(dense_single, 2)))
	print("{:<14}{:<11}{:<11}".format('eval_1', round(sparse_eval_1, 2), round(dense_eval_1, 2)))
	print("{:<14}{:<11}{:<11}".format('eval_2', round(sparse_eval_2, 2), round(dense_eval_2, 2)))
	print("{:<14}{:<11}{:<11}".format('eval_3', round(sparse_eval_3, 2), round(dense_eval_3, 2)))
	print("{:<14}{:<11}{:<11}".format('eval_4', round(sparse_eval_4, 2), round(dense_eval_4, 2)))
	print("{:<14}{:<11}{:<11}".format('eval_5', round(sparse_eval_5, 2), round(dense_eval_5, 2)))
	print("{:<14}{:<11}{:<11}".format('eval_AVG', round(sparse_AVG, 2), round(dense_AVG, 2)))
	

	
	