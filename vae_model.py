# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A class for loading trained MusicVAE models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import re
import tarfile

from backports import tempfile
import tensorflow as tf


class VAEModel(object):
	"""An interface to a trained model for encoding, decoding, and sampling.

	Args:
		config: The Config to build the model graph with.
		batch_size: The batch size to build the model graph with.
		checkpoint_dir_or_path: The directory containing checkpoints for the model,
			the most recent of which will be loaded, or a direct path to a specific
			checkpoint.
		var_name_substitutions: Optional list of string pairs containing regex
			patterns and substitution values for renaming model variables to match
			those in the checkpoint. Useful for backwards compatibility.
		session_target: Optional execution engine to connect to. Defaults to
			in-process.
		sample_kwargs: Additional, non-tensor keyword arguments to pass to sample
			call.
	"""
	
	def __init__(
			self, config, batch_size, checkpoint_dir_or_path=None,
			var_name_substitutions=None, session_target='', **sample_kwargs):
		if tf.gfile.IsDirectory(checkpoint_dir_or_path):
			checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir_or_path)
		else:
			checkpoint_path = checkpoint_dir_or_path
		self._config = copy.deepcopy(config)
		self._config.data_converter.set_mode('infer')
		self._config.hparams.batch_size = batch_size
		graph = tf.Graph()
		self._graph = graph
		with graph.as_default():
			model = self._config.model
			model.build(
				self._config.hparams,
				self._config.data_converter.output_depth,
				is_training=False)
			
			# Input placeholders
			self._temperature = tf.placeholder(tf.float32, shape=())
			
			if self._config.hparams.z_size:
				self._z_input = tf.placeholder(
					tf.float32, shape=[batch_size, self._config.hparams.z_size])
			else:
				self._z_input = None
			
			if self._config.data_converter.control_depth > 0:
				self._c_input = tf.placeholder(
					tf.float32, shape=[None, self._config.data_converter.control_depth])
			else:
				self._c_input = None
			
			self._inputs = tf.placeholder(
				tf.float32,
				shape=[batch_size, None, self._config.data_converter.input_depth])
			self._controls = tf.placeholder(
				tf.float32,
				shape=[batch_size, None, self._config.data_converter.control_depth])
			self._inputs_length = tf.placeholder(
				tf.int32,
				shape=[batch_size] + list(self._config.data_converter.length_shape))
			self._max_length = tf.placeholder(tf.int32, shape=())
			# Outputs
			self._outputs, self._decoder_results = model.sample(
				batch_size,
				max_length=self._max_length,
				z=self._z_input,
				c_input=self._c_input,
				temperature=self._temperature,
				**sample_kwargs)
			if self._config.hparams.z_size:
				q_z = model.encode(self._inputs, self._inputs_length, self._controls)
				self._mu = q_z.loc
				self._sigma = q_z.scale.diag
				self._z = q_z.sample()
			
			var_map = None
			
			if var_name_substitutions is not None:
				var_map = {}
				for v in tf.global_variables():
					var_name = v.name[:-2]  # Strip ':0' suffix.
					for pattern, substitution in var_name_substitutions:
						var_name = re.sub(pattern, substitution, var_name)
					if var_name != v.name[:-2]:
						tf.logging.info('Renaming `%s` to `%s`.', v.name[:-2], var_name)
					var_map[var_name] = v
			
			# Restore graph
			self._sess = tf.Session(target=session_target)
			saver = tf.train.Saver(var_map)
			if os.path.exists(checkpoint_path) and tarfile.is_tarfile(checkpoint_path):
				tf.logging.info('Unbundling checkpoint.')
				with tempfile.TemporaryDirectory() as temp_dir:
					tar = tarfile.open(checkpoint_path)
					tar.extractall(temp_dir)
					# Assume only a single checkpoint is in the directory.
					for name in tar.getnames():
						if name.endswith('.index'):
							checkpoint_path = os.path.join(temp_dir, name[0:-6])
							break
					saver.restore(self._sess, checkpoint_path)
			else:
				saver.restore(self._sess, checkpoint_path)
	
	@property
	def graph(self):
		return self._graph
	
	@property
	def sess(self):
		return self._sess
