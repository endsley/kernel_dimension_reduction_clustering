#!/usr/bin/env python3

import sklearn.metrics
import numpy as np
from sklearn.preprocessing import normalize			# version : 0.17




def Y_2_allocation(Y):
	i = 0
	allocation = np.array([])
	for m in range(Y.shape[0]):
		allocation = np.hstack((allocation, np.where(Y[m] == 1)[0][0]))
		i += 1

	return allocation


def Allocation_2_Y(allocation):
	
	N = np.size(allocation)
	unique_elements = np.unique(allocation)
	num_of_classes = len(unique_elements)
	class_ids = np.arange(num_of_classes)

	i = 0
	Y = np.zeros(num_of_classes)
	for m in allocation:
		class_label = np.where(unique_elements == m)[0]
		a_row = np.zeros(num_of_classes)
		a_row[class_label] = 1
		Y = np.hstack((Y, a_row))

	Y = np.reshape(Y, (N+1,num_of_classes))
	Y = np.delete(Y, 0, 0)

	return Y

def Kx_D_given_W(db, setX=None, setW=None):
	if setX is None: outX = db['Dloader'].X.dot(db['W'])
	else: outX = setX.dot(db['W'])
	
	if setW is None: outX = db['Dloader'].X.dot(db['W'])
	else: outX = db['Dloader'].X.dot(setW)

	if db['kernel_type'] == 'rbf':
		Kx = rbk_sklearn(outX, db['Dloader'].σ)
	elif db['kernel_type'] == 'rbf_slow':
		Kx = rbk_sklearn(outX, db['Dloader'].σ)
	elif db['kernel_type'] == 'linear':
		Kx = outX.dot(outX.T)
	elif db['kernel_type'] == 'polynomial':
		poly_sklearn(outX, db['poly_power'], db['poly_constant'])


	np.fill_diagonal(Kx, 0)			#	Set diagonal of adjacency matrix to 0
	D = compute_inverted_Degree_matrix(Kx)
	return [Kx, D]


def poly_sklearn(data, p, c):
	poly = sklearn.metrics.pairwise.polynomial_kernel(data, degree=p, coef0=c)
	return poly

def normalized_rbk_sklearn(X, σ):
	Kx = rbk_sklearn(X, σ)       	
	D = compute_inverted_Degree_matrix(Kx)
	return D.dot(Kx).dot(D)

def rbk_sklearn(data, sigma):
	gammaV = 1.0/(2*sigma*sigma)
	rbk = sklearn.metrics.pairwise.rbf_kernel(data, gamma=gammaV)
	np.fill_diagonal(rbk, 0)			#	Set diagonal of adjacency matrix to 0
	return rbk

def Ku_kernel(labels):
	Y = Allocation_2_Y(labels)
	Ky = Y.dot(Y.T)
	
	return Ky

def double_center(M, H):
	HMH = H.dot(M).dot(H)
	return HMH

def nomalized_by_Degree_matrix(M, D):
	D2 = np.diag(D)
	DMD = M*(np.outer(D2, D2))
	return DMD

def compute_inverted_Degree_matrix(M):
	return np.diag(1.0/np.sqrt(M.sum(axis=1)))

def compute_Degree_matrix(M):
	return np.diag(np.sum(M, axis=0))


def normalize_U(U):
	return normalize(U, norm='l2', axis=1)


def eig_solver(L, k, mode='smallest'):
	#L = ensure_matrix_is_numpy(L)
	eigenValues,eigenVectors = np.linalg.eigh(L)

	if mode == 'smallest':
		U = eigenVectors[:, 0:k]
		U_λ = eigenValues[0:k]
	elif mode == 'largest':
		n2 = len(eigenValues)
		n1 = n2 - k
		U = eigenVectors[:, n1:n2]
		U_λ = eigenValues[n1:n2]
	else:
		raise ValueError('unrecognized mode : ' + str(mode) + ' found.')
	
	return [U, U_λ]

