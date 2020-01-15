import os
import json
import tensorflow as tf
from keras import backend as K
from experiment import Experiment

class ExperimentSet:
	"""
	Set of experiments that can be executed sequentially.
	"""
	def __init__(self, json_path):
		self._json_path = json_path


	def _generate_experiments(self):
		# Load JSON file
		with open(self._json_path) as f:
			configs = json.load(f)

		# Add experiments
		for config in configs:
			val_type = config['val_type'] if 'val_type' in config else 'holdout'
			if val_type == 'holdout' and 'executions' in config:
				executions = int(config['executions'])
			elif val_type == 'kfold' and 'n_folds' in config:
				executions = int(config['n_folds'])
			else:
				raise Exception(F"{val_type} is not a valid validation type.")

			for execution in range(0, executions):
				exec_config = config.copy()
				if 'name' in exec_config:
					exec_config['name'] += "_{}".format(execution)
				exec_config['checkpoint_dir'] += "/{}".format(execution)
				experiment = Experiment()
				experiment.current_fold = execution
				experiment.set_config(exec_config)
				yield experiment


	def run_all(self, gpu_number=0):
		"""
		Execute all the experiments
		:param gpu_number: GPU that will be used.
		:return: None
		"""
		for experiment in self._generate_experiments():
			os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
			with tf.device('/device:GPU:' + str(gpu_number)):
				if not experiment.finished and experiment.task != 'test': # 'train' or 'both'
					experiment.run()
				if experiment.task != 'train': # 'test' or 'both'
					experiment.evaluate()
			# Clear session
			K.clear_session()