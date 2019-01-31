#!/usr/bin/env python3

from test_base import *
import sklearn.metrics
import numpy as np
from subprocess import call
from np_loader import *
from linear_unsupv_dim_reduction import *
from grassman import *


class test_obj(test_base):
	def __init__(self):
		db = {}
		db['data_name'] = 'car'
		db['data_source'] = 'numpy_files'				# link_download, load_image, local_file

		db['data_loader'] = np_loader
		db['TF_obj'] = linear_unsupv_dim_reduction
		db['W_optimize_technique'] = ism		# orthogonal_optimization, ism, DimGrowth, grassman
		db['compute_error'] = None
		db['store_results'] = None
		db['run_only_validation'] = True

		db['q'] = 4
		db['num_of_clusters'] = 4
		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma


		#db['run_debug_1'] = True
		#db['run_debug_2'] = True
		#db['run_debug_3'] = True

		test_base.__init__(self, db)

	def parameter_ranges(self):
		W_optimize_technique = [ism, DimGrowth, orthogonal_optimization]	
		return [W_optimize_technique]

prog = test_obj()
prog.basic_run()
#prog.batch_run()

