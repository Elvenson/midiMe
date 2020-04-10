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


class TrainedModel(object):
	"""An interface to a trained model for encoding, decoding, and sampling.

	Args:
		config: The Config to build the model graph with.
		batch_size: The batch size to build the model graph with.
		vae_checkpoint_dir_or_path: The directory containing VAE checkpoints for the model,
			the most recent of which will be loaded, or a direct path to a specific
			checkpoint.
		lc_vae_checkpoint_dir_or_path: The directory containing LC VAE checkpoints for the model,
			the most recent of which will be loaded, or a direct path to a specific
			checkpoint.
		lc_vae_var_pattern: List of string containing varible name pattern for LC VAE part
			patterns and substitution values for renaming model variables to match
			those in the checkpoint. Useful for backwards compatibility.
		session_target: Optional execution engine to connect to. Defaults to
			in-process.
		sample_kwargs: Additional, non-tensor keyword arguments to pass to sample
			call.
	"""
	
	def __init__(
			self, config, batch_size, vae_checkpoint_dir_or_path=None, lc_vae_checkpoint_dir_or_path=None,
			lc_vae_var_pattern=None, session_target='', **sample_kwargs):
		if tf.gfile.IsDirectory(vae_checkpoint_dir_or_path):
			vae_checkpoint_path = tf.train.latest_checkpoint(vae_checkpoint_dir_or_path)
		else:
			vae_checkpoint_path = vae_checkpoint_dir_or_path
			
		if tf.gfile.IsDirectory(lc_vae_checkpoint_dir_or_path):
			lc_vae_checkpoint_path = tf.train.latest_checkpoint(lc_vae_checkpoint_dir_or_path)
		else:
			lc_vae_checkpoint_path = lc_vae_checkpoint_dir_or_path
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
				encoder_train=False,
				decoder_train=False,
			)
			
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
			
			if lc_vae_var_pattern is None or len(lc_vae_var_pattern) == 0:
				raise ValueError("LC VAE variable pattern must be a non-empty list")
			
			vae_var_list = []
			lc_vae_var_list = []
			for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
				flag = False
				for pattern in lc_vae_var_pattern:
					if re.search(pattern, v.name):
						flag = True
						lc_vae_var_list.append(v)
				if not flag:
					vae_var_list.append(v)
			
			# Restore vae graph part
			self._sess = tf.Session(target=session_target)
			vae_saver = tf.train.Saver(vae_var_list)
			if os.path.exists(vae_checkpoint_path) and tarfile.is_tarfile(vae_checkpoint_path):
				tf.logging.info('Unbundling vae checkpoint.')
				with tempfile.TemporaryDirectory() as temp_dir:
					tar = tarfile.open(vae_checkpoint_path)
					tar.extractall(temp_dir)
					# Assume only a single checkpoint is in the directory.
					for name in tar.getnames():
						if name.endswith('.index'):
							vae_checkpoint_path = os.path.join(temp_dir, name[0:-6])
							break
					vae_saver.restore(self._sess, vae_checkpoint_path)
			else:
				vae_saver.restore(self._sess, vae_checkpoint_path)
				
			# Restore lc vae graph part
			lc_vae_saver = tf.train.Saver(lc_vae_var_list)
			if os.path.exists(vae_checkpoint_path) and tarfile.is_tarfile(lc_vae_checkpoint_path):
				tf.logging.info('Unbundling lc vae checkpoint.')
				with tempfile.TemporaryDirectory() as temp_dir:
					tar = tarfile.open(vae_checkpoint_path)
					tar.extractall(temp_dir)
					# Assume only a single checkpoint is in the directory.
					for name in tar.getnames():
						if name.endswith('.index'):
							lc_vae_checkpoint_path = os.path.join(temp_dir, name[0:-6])
							break
					lc_vae_saver.restore(self._sess, lc_vae_checkpoint_path)
			else:
				lc_vae_saver.restore(self._sess, lc_vae_checkpoint_path)
