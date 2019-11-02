import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

class C:
	def __init__(self, label):
		self.label = label
		self.data = []
		self.mean = 0
		self.cov_matrix = 0
		self.m = 0
		self.S = 0
	def add_data(self, new_sample):
		self.data.append(new_sample)
	def set_class_parameters(self):
		# Convert to a numpy array for computational convenience.
		self.data = np.array(self.data)
		# Compute the class mean.
		self.mean = np.mean(self.data, axis=0)
		# Compute class covariance matrix.
		# Pass the transpose of the data.
		self.cov_matrix = np.cov(self.data.T)
	def set_FLD_parameters(self, H):
		self.m = H.transpose() @ self.mean
		self.S = H.transpose() @ self.cov_matrix @ H


class Fisher_LDF:
	def __init__(self, classes):
		self.classes = classes
		self.A = self.compute_A()
		self.B = self.compute_B()
		self.H = self.compute_H()
		self.transform_classes()
	
	def compute_A(self):
		print ('Computing A matrix.')
		A = np.zeros(self.classes[0].cov_matrix.shape)
		for c in self.classes:
			A += c.cov_matrix
		return A
	
	def compute_B(self):
		print ('Computing B matrix.')
		overall_mean = 0.0
		for c in self.classes:
			overall_mean += c.mean
		overall_mean /= len(self.classes)
		feature_size = overall_mean.shape[0]
		# The shape of B is feature_size x feature_size
		B = np.zeros((feature_size, feature_size))
		for c in self.classes:
			T = c.mean - overall_mean
			# Add the multiplication of T vector and its transpose to the B. 
			B += T @ T.transpose()
		return B
	
	def compute_H(self):
		print ('Computing H matrix from eigenvectors.')
		temp = np.linalg.inv(self.A) @ self.B
		_, eigenvectors = np.linalg.eig(temp) 
		return eigenvectors
	
	def Mahalonobis_distance(self, c, test_data):
		f = self.transform_sample(test_data)
		temp = f - c.m 
		distance = temp.transpose() @ np.linalg.inv(c.S) @ temp
		return distance 

	def decide_class(self, sample):
		distances = []
		for c in self.classes:
			dist = self.Mahalonobis_distance(c, sample)
			distances.append(dist)

		# Return the index, ie. the class number, of the minimum distance.
		return distances.index(min(distances))

	def transform_sample(self, sample):
		return self.H.transpose() @ sample

	def transform_classes(self):
		print('Transforming classes to Fisher LDF space.')
		for c in self.classes:
			c.set_FLD_parameters(self.H)


#############################################################

def evaluation_from_cm(cm):
	FP = cm.sum(axis=0) - np.diag(cm)
	FN = cm.sum(axis=1) - np.diag(cm)
	TP = np.diag(cm)
	TN = cm.sum() - (FP + FN + TP)
	# Sensitivity, hit rate, recall, or true positive rate
	TPR = TP/(TP+FN)
	# Specificity or true negative rate
	TNR = TN/(TN+FP)
	# Precision or positive predictive value
	# PPV = TP/(TP+FP)
	# Negative predictive value
	# NPV = TN/(TN+FN)
	# Fall out or false positive rate
	FPR = FP/(FP+TN)
	# False negative rate
	FNR = FN/(TP+FN)
	# False discovery rate
	# FDR = FP/(TP+FP)
	# Overall accuracy
	ACC = (TP+TN)/(TP+FP+FN+TN)

	return ACC, TPR, TNR, FPR, FNR

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(file):
	data_dict = unpickle(file)
	data = data_dict[b'data']
	label = data_dict[b'labels']
	return data, label

def load_batches(file, batch_count):
	print('Loading data in batch 1')
	data, label = load_data(file+'1')
	for i in range(1, batch_count):
		filename = file + str(i+1)
		print('Loading data in batch', i+1)
		batch_data, batch_label = load_data(filename)
		data = np.concatenate([data, batch_data], axis=0)
		label.extend(batch_label)
	return data, label

def perform_test(fldf, test_set):
	print ('Performing tests.')
	decision_labels = []
	for i,test_sample in enumerate(test_set):
		decision = fldf.decide_class(test_sample)
		decision_labels.append(decision)
		# if (i+1)%10 == 0:
		print ('DONE: {0}/{1}'.format(i+1,len(test_set)))
	return decision_labels

#############################################################

# Create class objects. The index of each class represents their label.
classes = []
for i in range(10):
	classes.append(C(i))

data, labels = load_batches("../data/data_batch_", 5)

# Put the train data to the corresponding class.
for i, label in enumerate(labels):
	new_sample = data[i]
	classes[label].add_data(new_sample)

# Computing class means and covariance matrices.
print ('Computing class means and covariance matrices.')
for c in classes:
	c.set_class_parameters()

# Create a Fisher Linear Discriminant Object.
fldf = Fisher_LDF(classes)

test_data, test_labels = load_data("../data/test_batch")

decision_labels = perform_test(fldf, test_data[:50])

cm = confusion_matrix(test_labels[:50], decision_labels)
print(cm)

acc, tpr, tnr, fpr, fnr =evaluation_from_cm(cm)
print ("Accuracy:", acc)
print ("True positive rate:", tpr)
print ("True negative rate:", tnr)
print ("False positive rate:", fpr)
print ("False negative rate:", fnr)
