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

"""LSTM-based encoders and decoders for LC-MusicVAE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from magenta.common import flatten_maybe_padded_sequences
import base_model
from magenta.models.music_vae import lstm_utils
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import core as layers_core
from tensorflow.python.util import nest

rnn = tf.contrib.rnn
seq2seq = tf.contrib.seq2seq

# ENCODERS


class LstmEncoder(base_model.BaseEncoder):
	"""Unidirectional LSTM Encoder."""

	@property
	def output_depth(self):
		return self._cell.output_size

	def build(self, hparams, is_training=True, name_or_scope='encoder'):
		if hparams.use_cudnn and hparams.residual_encoder:
			raise ValueError('Residual connections not supported in cuDNN.')

		self._is_training = is_training
		self._name_or_scope = name_or_scope
		self._use_cudnn = hparams.use_cudnn

		tf.logging.info('\nEncoder Cells (unidirectional):\n'
										'  units: %s\n',
										hparams.enc_rnn_size)
		if self._use_cudnn:
			self._cudnn_lstm = lstm_utils.cudnn_lstm_layer(
					hparams.enc_rnn_size,
					hparams.dropout_keep_prob,
					is_training,
					name_or_scope=self._name_or_scope)
		else:
			self._cell = lstm_utils.rnn_cell(
					hparams.enc_rnn_size, hparams.dropout_keep_prob,
					hparams.residual_encoder, is_training)

	def encode(self, sequence, sequence_length):
		# Convert to time-major.
		sequence = tf.transpose(sequence, [1, 0, 2])
		if self._use_cudnn:
			outputs, _ = self._cudnn_lstm(
					sequence, training=self._is_training)
			return lstm_utils.get_final(outputs, sequence_length)
		else:
			outputs, _ = tf.nn.dynamic_rnn(
					self._cell, sequence, sequence_length, dtype=tf.float32,
					time_major=True, scope=self._name_or_scope)
			return outputs[-1]


class BidirectionalLstmEncoder(base_model.BaseEncoder):
	"""Bidirectional LSTM Encoder."""

	@property
	def output_depth(self):
		if self._use_cudnn:
			return self._cells[0][-1].num_units + self._cells[1][-1].num_units
		return self._cells[0][-1].output_size + self._cells[1][-1].output_size

	def build(self, hparams, is_training=True, name_or_scope='encoder'):
		if hparams.use_cudnn and hparams.residual_decoder:
			raise ValueError('Residual connections not supported in cuDNN.')

		self._is_training = is_training
		self._name_or_scope = name_or_scope
		self._use_cudnn = hparams.use_cudnn

		tf.logging.info('\nEncoder Cells (bidirectional):\n'
										'  units: %s\n',
										hparams.enc_rnn_size)

		if isinstance(name_or_scope, tf.VariableScope):
			name = name_or_scope.name
			reuse = name_or_scope.reuse
		else:
			name = name_or_scope
			reuse = None

		cells_fw = []
		cells_bw = []
		for i, layer_size in enumerate(hparams.enc_rnn_size):
			if self._use_cudnn:
				cells_fw.append(lstm_utils.cudnn_lstm_layer(
						[layer_size], hparams.dropout_keep_prob, is_training,
						name_or_scope=tf.VariableScope(
								reuse,
								name + '/cell_%d/bidirectional_rnn/fw' % i)))
				cells_bw.append(lstm_utils.cudnn_lstm_layer(
						[layer_size], hparams.dropout_keep_prob, is_training,
						name_or_scope=tf.VariableScope(
								reuse,
								name + '/cell_%d/bidirectional_rnn/bw' % i)))
			else:
				cells_fw.append(
						lstm_utils.rnn_cell(
								[layer_size], hparams.dropout_keep_prob,
								hparams.residual_encoder, is_training))
				cells_bw.append(
						lstm_utils.rnn_cell(
								[layer_size], hparams.dropout_keep_prob,
								hparams.residual_encoder, is_training))

		self._cells = (cells_fw, cells_bw)

	def encode(self, sequence, sequence_length):
		cells_fw, cells_bw = self._cells
		if self._use_cudnn:
			# Implements stacked bidirectional LSTM for variable-length sequences,
			# which are not supported by the CudnnLSTM layer.
			inputs_fw = tf.transpose(sequence, [1, 0, 2])
			for lstm_fw, lstm_bw in zip(cells_fw, cells_bw):
				outputs_fw, _ = lstm_fw(inputs_fw, training=self._is_training)
				inputs_bw = tf.reverse_sequence(
						inputs_fw, sequence_length, seq_axis=0, batch_axis=1)
				outputs_bw, _ = lstm_bw(inputs_bw, training=self._is_training)
				outputs_bw = tf.reverse_sequence(
						outputs_bw, sequence_length, seq_axis=0, batch_axis=1)

				inputs_fw = tf.concat([outputs_fw, outputs_bw], axis=2)

			last_h_fw = lstm_utils.get_final(outputs_fw, sequence_length)
			# outputs_bw has already been reversed, so we can take the first element.
			last_h_bw = outputs_bw[0]

		else:
			_, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
					cells_fw,
					cells_bw,
					sequence,
					sequence_length=sequence_length,
					time_major=False,
					dtype=tf.float32,
					scope=self._name_or_scope)
			# Note we access the outputs (h) from the states since the backward
			# ouputs are reversed to the input order in the returned outputs.
			last_h_fw = states_fw[-1][-1].h
			last_h_bw = states_bw[-1][-1].h

		return tf.concat([last_h_fw, last_h_bw], 1)


# DECODERS


class BaseLstmDecoder(base_model.BaseDecoder):
	"""Abstract LSTM Decoder class.

	Implementations must define the following abstract methods:
			-`_sample`
			-`_flat_reconstruction_loss`
	"""

	def build(self, hparams, output_depth, is_training=True):
		if hparams.use_cudnn and hparams.residual_decoder:
			raise ValueError('Residual connections not supported in cuDNN.')

		self._is_training = is_training
		self._z_size = hparams.z_size
		self._encoded_z_size = hparams.encoded_z_size

		tf.logging.info('\nDecoder Cells:\n'
										'  units: %s\n',
										hparams.dec_rnn_size)

		self._sampling_probability = lstm_utils.get_sampling_probability(
				hparams, is_training)
		self._output_depth = output_depth
		self._output_layer = layers_core.Dense(
				output_depth, name='output_projection')
		self._dec_cell = lstm_utils.rnn_cell(
				hparams.dec_rnn_size, hparams.dropout_keep_prob,
				hparams.residual_decoder, is_training)
		if hparams.use_cudnn:
			self._cudnn_dec_lstm = lstm_utils.cudnn_lstm_layer(
					hparams.dec_rnn_size, hparams.dropout_keep_prob, is_training,
					name_or_scope='decoder')
		else:
			self._cudnn_dec_lstm = None

	@property
	def state_size(self):
		return self._dec_cell.state_size

	@abc.abstractmethod
	def _sample(self, rnn_output, temperature):
		"""Core sampling method for a single time step.

		Args:
			rnn_output: The output from a single timestep of the RNN, sized
					`[batch_size, rnn_output_size]`.
			temperature: A scalar float specifying a sampling temperature.
		Returns:
			A batch of samples from the model.
		"""
		pass

	@abc.abstractmethod
	def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
		"""Core loss calculation method for flattened outputs.

		Args:
			flat_x_target: The flattened ground truth vectors, sized
				`[sum(x_length), self._output_depth]`.
			flat_rnn_output: The flattened output from all timeputs of the RNN,
				sized `[sum(x_length), rnn_output_size]`.
		Returns:
			r_loss: The unreduced reconstruction losses, sized `[sum(x_length)]`.
			metric_map: A map of metric names to tuples, each of which contain the
				pair of (value_tensor, update_op) from a tf.metrics streaming metric.
		"""
		pass

	def _decode(self, z, helper, input_shape, max_length=None):
		"""Decodes the given batch of latent vectors vectors, which may be 0-length.

		Args:
			z: Batch of latent vectors, sized `[batch_size, z_size]`, where `z_size`
				may be 0 for unconditioned decoding.
			helper: A seq2seq.Helper to use. If a TrainingHelper is passed and a
				CudnnLSTM has previously been defined, it will be used instead.
			input_shape: The shape of each model input vector passed to the decoder.
			max_length: (Optional) The maximum iterations to decode.

		Returns:
			results: The LstmDecodeResults.
		"""
		z_shape = z.shape[1].value
		if z_shape == self._z_size:
			name = 'decoder/z_to_initial_state'
		else:
			name = 'decoder/encoded_z_to_initial_state'
		initial_state = lstm_utils.initial_cell_state_from_embedding(
				self._dec_cell, z, name=name)

		# CudnnLSTM does not support sampling so it can only replace TrainingHelper.
		if self._cudnn_dec_lstm and type(helper) is seq2seq.TrainingHelper:  # pylint:disable=unidiomatic-typecheck
			rnn_output, _ = self._cudnn_dec_lstm(
					tf.transpose(helper.inputs, [1, 0, 2]),
					initial_state=lstm_utils.state_tuples_to_cudnn_lstm_state(
							initial_state),
					training=self._is_training)
			with tf.variable_scope('decoder'):
				rnn_output = self._output_layer(rnn_output)

			results = lstm_utils.LstmDecodeResults(
					rnn_input=helper.inputs[:, :, :self._output_depth],
					rnn_output=tf.transpose(rnn_output, [1, 0, 2]),
					samples=tf.zeros([z.shape[0], 0]),
					# TODO(adarob): Pass the final state when it is valid (fixed-length).
					final_state=None,
					final_sequence_lengths=helper.sequence_length)
		else:
			if self._cudnn_dec_lstm:
				tf.logging.warning(
						'CudnnLSTM does not support sampling. Using `dynamic_decode` '
						'instead.')
			decoder = lstm_utils.Seq2SeqLstmDecoder(
					self._dec_cell,
					helper,
					initial_state=initial_state,
					input_shape=input_shape,
					output_layer=self._output_layer)
			final_output, final_state, final_lengths = seq2seq.dynamic_decode(
					decoder,
					maximum_iterations=max_length,
					swap_memory=True,
					scope='decoder')
			results = lstm_utils.LstmDecodeResults(
					rnn_input=final_output.rnn_input[:, :, :self._output_depth],
					rnn_output=final_output.rnn_output,
					samples=final_output.sample_id,
					final_state=final_state,
					final_sequence_lengths=final_lengths)

		return results

	def reconstruction_loss(self, x_input, x_target, x_length, z=None,
													c_input=None):
		"""Reconstruction loss calculation.

		Args:
			x_input: Batch of decoder input sequences for teacher forcing, sized
				`[batch_size, max(x_length), output_depth]`.
			x_target: Batch of expected output sequences to compute loss against,
				sized `[batch_size, max(x_length), output_depth]`.
			x_length: Length of input/output sequences, sized `[batch_size]`.
			z: (Optional) Latent vectors. Required if model is conditional. Sized
				`[n, z_size]`.
			c_input: (Optional) Batch of control sequences, sized
					`[batch_size, max(x_length), control_depth]`. Required if conditioning
					on control sequences.

		Returns:
			r_loss: The reconstruction loss for each sequence in the batch.
			metric_map: Map from metric name to tf.metrics return values for logging.
			decode_results: The LstmDecodeResults.
		"""
		batch_size = x_input.shape[0].value

		has_z = z is not None
		z = tf.zeros([batch_size, 0]) if z is None else z
		repeated_z = tf.tile(
				tf.expand_dims(z, axis=1), [1, tf.shape(x_input)[1], 1])

		has_control = c_input is not None
		if c_input is None:
			c_input = tf.zeros([batch_size, tf.shape(x_input)[1], 0])

		sampling_probability_static = tensor_util.constant_value(
				self._sampling_probability)
		if sampling_probability_static == 0.0:
			# Use teacher forcing.
			x_input = tf.concat([x_input, repeated_z, c_input], axis=2)
			helper = seq2seq.TrainingHelper(x_input, x_length)
		else:
			# Use scheduled sampling.
			if has_z or has_control:
				auxiliary_inputs = tf.zeros([batch_size, tf.shape(x_input)[1], 0])
				if has_z:
					auxiliary_inputs = tf.concat([auxiliary_inputs, repeated_z], axis=2)
				if has_control:
					auxiliary_inputs = tf.concat([auxiliary_inputs, c_input], axis=2)
			else:
				auxiliary_inputs = None
			helper = seq2seq.ScheduledOutputTrainingHelper(
					inputs=x_input,
					sequence_length=x_length,
					auxiliary_inputs=auxiliary_inputs,
					sampling_probability=self._sampling_probability,
					next_inputs_fn=self._sample)

		decode_results = self._decode(
				z, helper=helper, input_shape=helper.inputs.shape[2:])
		flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
		flat_rnn_output = flatten_maybe_padded_sequences(
				decode_results.rnn_output, x_length)
		r_loss, metric_map = self._flat_reconstruction_loss(
				flat_x_target, flat_rnn_output)

		# Sum loss over sequences.
		cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)
		r_losses = []
		for i in range(batch_size):
			b, e = cum_x_len[i], cum_x_len[i + 1]
			r_losses.append(tf.reduce_sum(r_loss[b:e]))
		r_loss = tf.stack(r_losses)

		return r_loss, metric_map, decode_results

	def sample(self, n, max_length=None, z=None, c_input=None, temperature=1.0,
						 start_inputs=None, end_fn=None):
		"""Sample from decoder with an optional conditional latent vector `z`.

		Args:
			n: Scalar number of samples to return.
			max_length: (Optional) Scalar maximum sample length to return. Required if
				data representation does not include end tokens.
			z: (Optional) Latent vectors to sample from. Required if model is
				conditional. Sized `[n, z_size]`.
			c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
			temperature: (Optional) The softmax temperature to use when sampling, if
				applicable.
			start_inputs: (Optional) Initial inputs to use for batch.
				Sized `[n, output_depth]`.
			end_fn: (Optional) A callable that takes a batch of samples (sized
				`[n, output_depth]` and emits a `bool` vector
				shaped `[batch_size]` indicating whether each sample is an end token.
		Returns:
			samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
			final_state: The final states of the decoder.
		Raises:
			ValueError: If `z` is provided and its first dimension does not equal `n`.
		"""
		if z is not None and z.shape[0].value != n:
			raise ValueError(
					'`z` must have a first dimension that equals `n` when given. '
					'Got: %d vs %d' % (z.shape[0].value, n))

		# Use a dummy Z in unconditional case.
		z = tf.zeros((n, 0), tf.float32) if z is None else z

		if c_input is not None:
			# Tile control sequence across samples.
			c_input = tf.tile(tf.expand_dims(c_input, 1), [1, n, 1])

		# If not given, start with zeros.
		if start_inputs is None:
			start_inputs = tf.zeros([n, self._output_depth], dtype=tf.float32)
		# In the conditional case, also concatenate the Z.
		start_inputs = tf.concat([start_inputs, z], axis=-1)
		if c_input is not None:
			start_inputs = tf.concat([start_inputs, c_input[0]], axis=-1)
		initialize_fn = lambda: (tf.zeros([n], tf.bool), start_inputs)

		sample_fn = lambda time, outputs, state: self._sample(outputs, temperature)
		end_fn = end_fn or (lambda x: False)

		def next_inputs_fn(time, outputs, state, sample_ids):
			del outputs
			finished = end_fn(sample_ids)
			next_inputs = tf.concat([sample_ids, z], axis=-1)
			if c_input is not None:
				next_inputs = tf.concat([next_inputs, c_input[time]], axis=-1)
			return (finished, next_inputs, state)

		sampler = seq2seq.CustomHelper(
				initialize_fn=initialize_fn, sample_fn=sample_fn,
				next_inputs_fn=next_inputs_fn, sample_ids_shape=[self._output_depth],
				sample_ids_dtype=tf.float32)

		decode_results = self._decode(
				z, helper=sampler, input_shape=start_inputs.shape[1:],
				max_length=max_length)

		return decode_results.samples, decode_results


class CategoricalLstmDecoder(BaseLstmDecoder):
	"""LSTM decoder with single categorical output."""

	def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
		flat_logits = flat_rnn_output
		flat_truth = tf.argmax(flat_x_target, axis=1)
		flat_predictions = tf.argmax(flat_logits, axis=1)
		r_loss = tf.nn.softmax_cross_entropy_with_logits(
				labels=flat_x_target, logits=flat_logits)

		metric_map = {
				'metrics/accuracy':
						tf.metrics.accuracy(flat_truth, flat_predictions),
				'metrics/mean_per_class_accuracy':
						tf.metrics.mean_per_class_accuracy(
								flat_truth, flat_predictions, flat_x_target.shape[-1].value),
		}
		return r_loss, metric_map

	def _sample(self, rnn_output, temperature=1.0):
		sampler = tfp.distributions.OneHotCategorical(
				logits=rnn_output / temperature, dtype=tf.float32)
		return sampler.sample()

	def sample(self, n, max_length=None, z=None, c_input=None, temperature=None,
						 start_inputs=None, beam_width=None, end_token=None):
		"""Overrides BaseLstmDecoder `sample` method to add optional beam search.

		Args:
			n: Scalar number of samples to return.
			max_length: (Optional) Scalar maximum sample length to return. Required if
				data representation does not include end tokens.
			z: (Optional) Latent vectors to sample from. Required if model is
				conditional. Sized `[n, z_size]`.
			c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
			temperature: (Optional) The softmax temperature to use when not doing beam
				search. Defaults to 1.0. Ignored when `beam_width` is provided.
			start_inputs: (Optional) Initial inputs to use for batch.
				Sized `[n, output_depth]`.
			beam_width: (Optional) Width of beam to use for beam search. Beam search
				is disabled if not provided.
			end_token: (Optional) Scalar token signaling the end of the sequence to
				use for early stopping.
		Returns:
			samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
			final_state: The final states of the decoder.
		Raises:
			ValueError: If `z` is provided and its first dimension does not equal `n`,
				or if `c_input` is provided under beam search.
		"""
		if beam_width is None:
			if end_token is None:
				end_fn = None
			else:
				end_fn = lambda x: tf.equal(tf.argmax(x, axis=-1), end_token)
			return super(CategoricalLstmDecoder, self).sample(
					n, max_length, z, c_input, temperature, start_inputs, end_fn)

		# TODO(iansimon): Support conditioning in beam search decoder, which may be
		# awkward as there's no helper.
		if c_input is not None:
			raise ValueError('Control sequence unsupported in beam search.')

		# If `end_token` is not given, use an impossible value.
		end_token = self._output_depth if end_token is None else end_token
		if z is not None and z.shape[0].value != n:
			raise ValueError(
					'`z` must have a first dimension that equals `n` when given. '
					'Got: %d vs %d' % (z.shape[0].value, n))

		if temperature is not None:
			tf.logging.warning('`temperature` is ignored when using beam search.')
		# Use a dummy Z in unconditional case.
		z = tf.zeros((n, 0), tf.float32) if z is None else z

		# If not given, start with dummy `-1` token and replace with zero vectors in
		# `embedding_fn`.
		if start_inputs is None:
			start_tokens = -1 * tf.ones([n], dtype=tf.int32)
		else:
			start_tokens = tf.argmax(start_inputs, axis=-1, output_type=tf.int32)

		initial_state = lstm_utils.initial_cell_state_from_embedding(
				self._dec_cell, z, name='decoder/z_to_initial_state')
		beam_initial_state = seq2seq.tile_batch(
				initial_state, multiplier=beam_width)

		# Tile `z` across beams.
		beam_z = tf.tile(tf.expand_dims(z, 1), [1, beam_width, 1])

		def embedding_fn(tokens):
			# If tokens are the start_tokens (negative), replace with zero vectors.
			next_inputs = tf.cond(
					tf.less(tokens[0, 0], 0),
					lambda: tf.zeros([n, beam_width, self._output_depth]),
					lambda: tf.one_hot(tokens, self._output_depth))

			# Concatenate `z` to next inputs.
			next_inputs = tf.concat([next_inputs, beam_z], axis=-1)
			return next_inputs

		decoder = seq2seq.BeamSearchDecoder(
				self._dec_cell,
				embedding_fn,
				start_tokens,
				end_token,
				beam_initial_state,
				beam_width,
				output_layer=self._output_layer,
				length_penalty_weight=0.0)

		final_output, final_state, final_lengths = seq2seq.dynamic_decode(
				decoder,
				maximum_iterations=max_length,
				swap_memory=True,
				scope='decoder')

		samples = tf.one_hot(final_output.predicted_ids[:, :, 0],
												 self._output_depth)
		# Rebuild the input by combining the inital input with the sampled output.
		if start_inputs is None:
			initial_inputs = tf.zeros([n, 1, self._output_depth])
		else:
			initial_inputs = tf.expand_dims(start_inputs, axis=1)

		rnn_input = tf.concat([initial_inputs, samples[:, :-1]], axis=1)

		results = lstm_utils.LstmDecodeResults(
				rnn_input=rnn_input,
				rnn_output=None,
				samples=samples,
				final_state=nest.map_structure(
						lambda x: x[:, 0], final_state.cell_state),
				final_sequence_lengths=final_lengths[:, 0])
		return samples, results


class MultiOutCategoricalLstmDecoder(CategoricalLstmDecoder):
	"""LSTM decoder with multiple categorical outputs.

	The final sequence dimension is split before computing the loss or sampling,
	based on the `output_depths`. Reconstruction losses are summed across the
	split and samples are concatenated in the same order as the input.

	Args:
		output_depths: A list of output depths for the in the same order as the are
			concatenated in the final sequence dimension.
	"""

	def __init__(self, output_depths):
		self._output_depths = output_depths

	def build(self, hparams, output_depth, is_training=True):
		if sum(self._output_depths) != output_depth:
			raise ValueError(
					'Decoder output depth does not match sum of sub-decoders: %s vs %d' %
					(self._output_depths, output_depth))
		super(MultiOutCategoricalLstmDecoder, self).build(
				hparams, output_depth, is_training)

	def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
		split_x_target = tf.split(flat_x_target, self._output_depths, axis=-1)
		split_rnn_output = tf.split(
				flat_rnn_output, self._output_depths, axis=-1)

		losses = []
		metric_map = {}
		for i in range(len(self._output_depths)):
			l, m = (
					super(MultiOutCategoricalLstmDecoder, self)._flat_reconstruction_loss(
							split_x_target[i], split_rnn_output[i]))
			losses.append(l)
			for k, v in m.items():
				metric_map['%s/output_%d' % (k, i)] = v

		return tf.reduce_sum(losses, axis=0), metric_map

	def _sample(self, rnn_output, temperature=1.0):
		split_logits = tf.split(rnn_output, self._output_depths, axis=-1)
		samples = []
		for logits, output_depth in zip(split_logits, self._output_depths):
			sampler = tfp.distributions.Categorical(
					logits=logits / temperature)
			sample_label = sampler.sample()
			samples.append(tf.one_hot(sample_label, output_depth, dtype=tf.float32))
		return tf.concat(samples, axis=-1)
	

def get_default_hparams():
	"""Returns copy of default HParams for LSTM models."""
	hparams_map = base_model.get_default_hparams().values()
	hparams_map.update({
		'conditional': True,
		'dec_rnn_size': [512],  # Decoder RNN: number of units per layer.
		'enc_rnn_size': [256],  # Encoder RNN: number of units per layer per dir.
		'dropout_keep_prob': 1.0,  # Probability all dropout keep.
		'sampling_schedule': 'constant',  # constant, exponential, inverse_sigmoid
		'sampling_rate': 0.0,  # Interpretation is based on `sampling_schedule`.
		'use_cudnn': False,  # Uses faster CudnnLSTM to train. For GPU only.
		'residual_encoder': False,  # Use residual connections in encoder.
		'residual_decoder': False,  # Use residual connections in decoder.
	})
	return tf.contrib.training.HParams(**hparams_map)
