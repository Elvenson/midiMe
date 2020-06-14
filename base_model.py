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

# Modification copyright 2020 Bui Quoc Bao.
# Add Latent Constraint VAE model.
# Add Small VAE model.

""" Base Latent Constraint MusicVAE model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from tensorflow.contrib import metrics as contrib_metrics

ds = tfp.distributions


class BaseEncoder(six.with_metaclass(abc.ABCMeta, object)):
    """
    Abstract encoder class.
    Implementations must define the following abstract methods:
        -`build`
        -`encode`
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def output_depth(self):
        """Returns the size of the output final dimension."""
        pass

    @abc.abstractmethod
    def build(self, hparams, is_training=True):
        """
        Builder method for BaseEncoder.
        :param hparams: An HParams object containing model hyperparameters.
        :param is_training: Whether or not the model is being used for training.
        """
        pass

    @abc.abstractmethod
    def encode(self, sequence, sequence_length):
        """
        Encodes input sequences into a precursors for latent code `z`.
        :param sequence: Batch of sequences to encode.
        :param sequence_length: Length of sequences in input batch.
        :return:
        outputs: Raw outputs to parameterize the prior distribution in MusicVae.encode,
         sized `[batch_size, N]`.
        """
        pass


class BaseDecoder(six.with_metaclass(abc.ABCMeta, object)):
    """
    Abstract decoder class.
    Implementations must define the following abstract methods:
        -`build`
        -`reconstruction_loss`
        -`sample`
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def build(self, hparams, output_depth, is_training=True):
        """
        Builder method for BaseDecoder.
        :param hparams: An HParams object containing model hyperparameters.
        :param output_depth: Size of final output dimension.
        :param is_training: Whether or not the model is being used for training.
        """
        pass

    @abc.abstractmethod
    def reconstruction_loss(self, x_input, x_target, x_length, z=None, c_input=None):
        """
        Reconstruction loss calculation.
        :param x_input: Batch of decoder input sequences for teacher forcing, sized
                    `[batch_size, max(x_length), output_depth]`.
        :param x_target: Batch of expected output sequences to compute loss against,
                    sized `[batch_size, max(x_length), output_depth]`.
        :param x_length: Length of input/output sequences, sized `[batch_size]`.
        :param z: (Optional) Latent vectors. Required if model is conditional. Sized
                    `[n, z_size]`.
        :param c_input: (Optional) Batch of control sequences, sized
                    `[batch_size, max(x_length), control_depth]`. Required if conditioning
                    on control sequences.

        :return:
            r_loss: The reconstruction loss for each sequence in the batch.
            metric_map: Map from metric name to tf.metrics return values for logging.
        """
        pass

    @abc.abstractmethod
    def sample(self, n, max_length=None, z=None, c_input=None):
        """
        Sample from decoder with an optional conditional latent vector `z`.
        :param	n: Scalar number of samples to return.
        :param	max_length: (Optional) Scalar maximum sample length to return. Required if
                data representation does not include end tokens.
        :param	z: (Optional) Latent vectors to sample from. Required if model is
                conditional. Sized `[n, z_size]`.
        :param	c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.

        :return:
            samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
        """
        pass


class SmallMusicVAE(object):
    """Small Latent Music Variation Autoencoder."""

    def __init__(self, encoder, decoder):
        """
        Initializer for a SmallMusicVAE model.
        :param encoder: A BaseEncoder implementation class to use.
        :param decoder: A BaseDecoder implementation class to use.
        """
        self._encoder = encoder
        self._decoder = decoder

    def build(self, hparams, output_depth, encoder_train=False, decoder_train=False):
        """
        Builds encoder and decoder.
        Must be called within a graph.
        :param encoder_train: Whether or not to train the encoder part
        :param decoder_train Whether or not to train the decoder part
        :param hparams: An HParams object containing model hyperparameters. See
            `get_default_hparams` below for required values.
        :param output_depth: Size of final output dimension.
        """
        tf.logging.info(
            'Building LCMusicVAE model with %s, %s, and hparams:\n%s',
            self.encoder.__class__.__name__,
            self.decoder.__class__.__name__,
            hparams.values())
        self.global_step = tf.train.get_or_create_global_step()
        self._hparams = hparams
        self._encoder.build(hparams, encoder_train)
        self._decoder.build(hparams, output_depth, decoder_train)

    @property
    def encoder(self):
        """Gets encoder."""
        return self._encoder

    @property
    def decoder(self):
        """Get decoder."""
        return self._decoder

    @property
    def hparams(self):
        """Get hparams."""
        return self._hparams

    def encode(self, sequence, sequence_length, control_sequence=None):
        """
        Encodes input sequences into a MultivariateNormalDiag distribution.
        :param sequence: A Tensor with shape `[num_sequences, max_length, input_depth]`
            containing the sequences to encode.
        :param sequence_length: The length of each sequence in the `sequence` Tensor.
        :param control_sequence: (Optional) A Tensor with shape
            `[num_sequences, max_length, control_length]` containing control sequences
            on which to condition. These will be concatenated depthwise to the input sequences.
        :return:
            A tfp.distribution.MultivariateNormalDiag representing the posterior distribution
            for each sequence.
        """
        hparams = self.hparams
        z_size = hparams.z_size

        sequence = tf.to_float(sequence)
        if control_sequence is not None:
            control_sequence = tf.to_float(control_sequence)
            sequence = tf.concat([sequence, control_sequence], axis=-1)
        encoder_output = self.encoder.encode(sequence, sequence_length)

        mu = tf.layers.Dense(
            z_size,
            name='encoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
        )(encoder_output)
        sigma = tf.layers.Dense(
            z_size,
            activation=tf.nn.softplus,
            name='encoder/sigma',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
        )(encoder_output)

        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def encode_latent(self, z):
        """
        Encodes z distribution into a smaller latent MultivariateNormalDiag distribution.
        :param z: A Tensor with shape `[batch_size, z_size]`
        :return:
            A tfp.distribution.MultivariateNormalDiag representing the smaller z latent space
        """
        hparams = self.hparams
        latent_encoder_layers = hparams.latent_encoder_layers
        encoded_z_size = hparams.encoded_z_size

        x = z
        for i, layer_size in enumerate(latent_encoder_layers):
            x = tf.layers.Dense(
                layer_size,
                name='latent_encoder/layer{}'.format(i),
                activation='relu'
            )(x)

        # Create smaller latent z' distribution
        mu = tf.layers.Dense(
            encoded_z_size,
            name='latent_encoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001)
        )(x)
        sigma = tf.layers.Dense(
            encoded_z_size,
            activation=tf.nn.softplus,
            name='latent_encoder/sigma',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001)
        )(x)

        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def _decode_latent(self, latent_z):
        """
        Decodes latent z distribution into z distribution
        :param latent_z: A Tensor with shape `[batch_size, encoded_z_size]`
        :return:
            A tfp.distribution.MultivariateNormalDiag reconstructing the z space
        """
        hparams = self.hparams
        latent_decoder_layers = hparams.latent_decoder_layers
        z_size = hparams.z_size

        x = latent_z
        for i, layer_size in enumerate(latent_decoder_layers):
            x = tf.layers.Dense(
                layer_size,
                name='latent_decoder/layer{}'.format(i),
                activation='relu'
            )(x)

        # Recreate z distribution
        mu = tf.layers.Dense(
            z_size,
            name='latent_decoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001)
        )(x)

        return mu

    def _latent_reconstruction_loss(self, z, latent_z):
        """
        Reconstruction loss between constructed z and z.
        :param z: A Tensor with shape `[batch_size, z_size]`
            containing real sample from z distribution
        :param latent_z: A Tensor with shape `[batch_size, encoded_z_size]`
            containing encoded `z`
        :return:
            Tensor represent reconstruction loss size [batch_size]
        """
        g_z = self._decode_latent(latent_z)

        # Loss is calculated by this formula
        # mse(x,input_mu=0) / (2 * input_sigma=1 ^ 2)
        se = tf.math.pow(g_z - z, 2)
        loss = tf.math.divide(se, tf.math.multiply(2.0, tf.math.pow(tf.ones([1]), 2)))
        return tf.reduce_sum(loss, -1)

    def _kl_latent_loss(self, encode_latent):
        """
        Kullback Leibler loss between latent z distribution with N(0,1)
        :param encode_latent: A tfp.distribution.MultivariateNormalDiag
         representing smaller latent `z` distribution
        :return:
            Kullback Leibler divergence between N(0,1) and encoded `z` distribution
        """
        hparams = self.hparams
        # Prior latent distribution
        p_latent_z = ds.MultivariateNormalDiag(
            loc=[0.] * hparams.encoded_z_size, scale_diag=[1.] * hparams.encoded_z_size
        )

        return ds.kl_divergence(encode_latent, p_latent_z)

    def _compute_model_loss(
            self, input_sequence, output_sequence, sequence_length, control_sequence):
        """Builds a model with loss for train/eval."""
        hparams = self.hparams
        batch_size = hparams.batch_size

        input_sequence = tf.to_float(input_sequence)
        output_sequence = tf.to_float(output_sequence)

        max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)

        input_sequence = input_sequence[:, :max_seq_len]

        if control_sequence is not None:
            control_depth = control_sequence.shape[-1]
            control_sequence = tf.to_float(control_sequence)
            control_sequence = control_sequence[:, :max_seq_len]
            # Shouldn't be necessary, but the slice loses shape information when
            # control depth is zero.
            control_sequence.set_shape([batch_size, None, control_depth])

        # Inputs to be fed to decoder, including zero padding for the initial input.
        x_length = tf.minimum(sequence_length, max_seq_len)
        q_z = self.encode(input_sequence, x_length, control_sequence)
        z = q_z.sample()
        q_encode_z = self.encode_latent(z)
        latent_z = q_encode_z.sample()

        # KL Divergence (nats)
        kl_cost = self._kl_latent_loss(q_encode_z)

        # Reconstruction loss
        r_loss = self._latent_reconstruction_loss(z, latent_z)

        beta = ((1.0 - tf.pow(hparams.beta_rate, tf.to_float(self.global_step))) * hparams.max_beta)
        self.loss = tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost)

        scalars_to_summarize = {
            'loss': self.loss,
            'losses/r_loss': r_loss,
            'losses/kl_loss': kl_cost,
            'losses/kl_beta': beta,
        }
        return scalars_to_summarize

    def train(self, input_sequence, output_sequence, sequence_length, control_sequence=None):
        """
        Train on the given sequences, returning an optimizer.
        :param input_sequence: The sequence to be fed to the encoder.
        :param output_sequence: The sequence expected from the decoder.
        :param sequence_length: The length of the given sequences (which must be
            identical).
        :param control_sequence: (Optional) sequence on which to condition. This will be
            concatenated depthwise to the model inputs for both encoding and decoding.
        :return:
            optimizer: A tf.train.Optimizer.
        """

        scalars_to_summarize = self._compute_model_loss(
            input_sequence, output_sequence, sequence_length, control_sequence
        )

        hparams = self.hparams
        lr = ((hparams.learning_rate - hparams.min_learning_rate) *
              tf.pow(hparams.decay_rate, tf.to_float(self.global_step)) +
              hparams.min_learning_rate)

        tf.summary.scalar('learning_rate', lr)
        for n, t in scalars_to_summarize.items():
            tf.summary.scalar(n, tf.reduce_mean(t))

        optimizer = tf.train.AdamOptimizer(lr)

        return optimizer

    def eval(self, input_sequence, output_sequence, sequence_length, control_sequence=None):
        """
        Evaluate on the given sequences, returning metric update ops.
        :param input_sequence: The sequence to be fed to the encoder.
        :param output_sequence: The sequence expected from the decoder.
        :param sequence_length: The length of the given sequences (which must be
            identical).
        :param control_sequence: (Optional) sequence on which to condition the decoder.
        :return:
            metric_update_ops: tf.metrics update ops.
        """
        scalars_to_summarize = self._compute_model_loss(
            input_sequence, output_sequence, sequence_length, control_sequence
        )

        metric_map = {}
        for n, t in scalars_to_summarize.items():
            metric_map[n] = tf.metrics.mean(t)

        metrics_to_values, metrics_to_update = (
            contrib_metrics.aggregate_metric_map(metric_map)
        )

        for metric_name, metric_value in metrics_to_values.items():
            tf.summary.scalar(metric_name, metric_value)

        return list(metrics_to_update.values())

    def sample(self, n, max_length=None, latent_z=None, c_input=None, **kwargs):
        """Sample with on optional conditional embedding `z`."""
        if latent_z is not None and latent_z.shape[0].value != n:
            raise ValueError(
                '`z` must have a first dimension that equals `n` when given. '
                'Got: %d vs %d' % (latent_z.shape[0].value, n)
            )
        if self.hparams.encoded_z_size and latent_z is None:
            tf.logging.warning(
                'Sampling from conditional model without `z`. Using random `z`.'
            )
            normal_shape = [n, self.hparams.encoded_z_size]
            normal_dist = tfp.distributions.Normal(
                loc=tf.zeros(normal_shape), scale=tf.ones(normal_shape)
            )
            latent_z = normal_dist.sample()

        z = self._decode_latent(latent_z)

        return self.decoder.sample(n, max_length, z, c_input, **kwargs)


class LCMusicVAE(object):
    """Latent Constraint Music Variational Autoencoder."""

    def __init__(self, encoder, decoder):
        """
        Initializer for a LCMusicVAE model.
        :param encoder: A BaseEncoder implementation class to use.
        :param decoder: A BaseDecoder implementation class to use.
        """
        self._encoder = encoder
        self._decoder = decoder

    def build(self, hparams, output_depth, encoder_train=False, decoder_train=True):
        """
        Builds encoder and decoder.
        Must be called within a graph.
        :param encoder_train: Whether or not to train the encoder part
        :param decoder_train Whether or not to train the decoder part
        :param hparams: An HParams object containing model hyperparameters. See
            `get_default_hparams` below for required values.
        :param output_depth: Size of final output dimension.
        """
        tf.logging.info(
            'Building LCMusicVAE model with %s, %s, and hparams:\n%s',
            self.encoder.__class__.__name__,
            self.decoder.__class__.__name__,
            hparams.values())
        self.global_step = tf.train.get_or_create_global_step()
        self._hparams = hparams
        self._encoder.build(hparams, encoder_train)
        self._decoder.build(hparams, output_depth, decoder_train)

    @property
    def encoder(self):
        """Gets encoder."""
        return self._encoder

    @property
    def decoder(self):
        """Get decoder."""
        return self._decoder

    @property
    def hparams(self):
        """Get hparams."""
        return self._hparams

    def encode(self, sequence, sequence_length, control_sequence=None):
        """
        Encodes input sequences into a MultivariateNormalDiag distribution.
        :param sequence: A Tensor with shape `[num_sequences, max_length, input_depth]`
            containing the sequences to encode.
        :param sequence_length: The length of each sequence in the `sequence` Tensor.
        :param control_sequence: (Optional) A Tensor with shape
            `[num_sequences, max_length, control_length]` containing control sequences
            on which to condition. These will be concatenated depthwise to the input sequences.
        :return:
            A tfp.distribution.MultivariateNormalDiag representing the posterior distribution
            for each sequence.
        """
        hparams = self.hparams
        z_size = hparams.z_size

        sequence = tf.to_float(sequence)
        if control_sequence is not None:
            control_sequence = tf.to_float(control_sequence)
            sequence = tf.concat([sequence, control_sequence], axis=-1)
        encoder_output = self.encoder.encode(sequence, sequence_length)

        mu = tf.layers.Dense(
            z_size,
            name='encoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
        )(encoder_output)
        sigma = tf.layers.Dense(
            z_size,
            activation=tf.nn.softplus,
            name='encoder/sigma',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
        )(encoder_output)

        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def encode_latent(self, z):
        """
        Encodes z distribution into a smaller latent MultivariateNormalDiag distribution.
        :param z: A Tensor with shape `[batch_size, z_size]`
        :return:
            A tfp.distribution.MultivariateNormalDiag representing the smaller z latent space
        """
        hparams = self.hparams
        encoded_z_size = hparams.encoded_z_size

        mu = tf.layers.Dense(
            encoded_z_size,
            name='latent_encoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001)
        )(z)
        sigma = tf.layers.Dense(
            encoded_z_size,
            activation=tf.nn.softplus,
            name='latent_encoder/sigma',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001)
        )(z)

        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def _decode_latent(self, latent_z):
        """
        Decodes latent z distribution into z distribution
        :param latent_z: A Tensor with shape `[batch_size, encoded_z_size]`
        :return:
            A tfp.distribution.MultivariateNormalDiag reconstructing the z space
        """
        hparams = self.hparams
        z_size = hparams.z_size

        mu = tf.layers.Dense(
            z_size,
            name='latent_decoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001)
        )(latent_z)
        sigma = tf.layers.Dense(
            z_size,
            activation=tf.nn.softplus,
            name='latent_decoder/sigma',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001)
        )(latent_z)

        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def _latent_reconstruction_loss(self, z, latent_z):
        """
        Reconstruction loss between constructed z and z.
        :param z: A Tensor with shape `[batch_size, z_size]`
            containing real sample from z distribution
        :param latent_z: A Tensor with shape `[batch_size, encoded_z_size]`
            containing encoded `z`
        :return:
            A L2 loss representing the reconstruction loss size [batch_size, 1]
        """
        g_z = self._decode_latent(latent_z)

        return tf.norm(
            g_z.sample() - z, ord='euclidean', axis=1, keepdims=True, name=None
        )

    def _kl_latent_loss(self, encode_latent):
        """
        Kullback Leibler loss between latent z distribution with N(0,1)
        :param encode_latent: A tfp.distribution.MultivariateNormalDiag
         representing smaller latent `z` distribution
        :return:
            Kullback Leibler divergence between N(0,1) and encoded `z` distribution
        """
        hparams = self.hparams
        # Prior latent distribution
        p_latent_z = ds.MultivariateNormalDiag(
            loc=[0.] * hparams.encoded_z_size, scale_diag=[1.] * hparams.encoded_z_size
        )

        return ds.kl_divergence(encode_latent, p_latent_z)

    def _compute_model_loss(
            self, input_sequence, output_sequence, sequence_length, control_sequence):
        """Builds a model with loss for train/eval."""
        hparams = self.hparams
        batch_size = hparams.batch_size

        input_sequence = tf.to_float(input_sequence)
        output_sequence = tf.to_float(output_sequence)

        max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)

        input_sequence = input_sequence[:, :max_seq_len]

        if control_sequence is not None:
            control_depth = control_sequence.shape[-1]
            control_sequence = tf.to_float(control_sequence)
            control_sequence = control_sequence[:, :max_seq_len]
            # Shouldn't be necessary, but the slice loses shape information when
            # control depth is zero.
            control_sequence.set_shape([batch_size, None, control_depth])

        # The target/expected outputs.
        x_target = output_sequence[:, :max_seq_len]
        # Inputs to be fed to decoder, including zero padding for the initial input.
        x_input = tf.pad(output_sequence[:, :max_seq_len - 1], [(0, 0), (1, 0), (0, 0)])
        x_length = tf.minimum(sequence_length, max_seq_len)

        # Either encode to get `z`, to get smaller latent `z` or do unconditional, decoder-only.
        if hparams.z_size:
            q_z = self.encode(input_sequence, x_length, control_sequence)
            z = q_z.sample()

            if hparams.encoded_z_size:  # lc-vae mode
                # Encode `z` into smaller latent space
                q_encode_z = self.encode_latent(z)
                latent_z = q_encode_z.sample()
   
                # KL Divergence (nats)
                kl_div = self._kl_latent_loss(q_encode_z)

                # L2 loss in smaller latent space
                l2 = self._latent_reconstruction_loss(z, latent_z)
            else:  # vae mode:
                # Prior distribution
                p_z = ds.MultivariateNormalDiag(
                    loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size
                )

                # KL Divergence (nats)
                kl_div = ds.kl_divergence(q_z, p_z)
                l2 = tf.zeros([batch_size, 1], dtype=tf.float32)
                latent_z = None
        else:  # unconditional, decoder-only generation
            kl_div = tf.zeros([batch_size, 1], dtype=tf.float32)
            l2 = tf.zeros([batch_size, 1], dtype=tf.float32)
            z = None
            latent_z = None

        # Concatenate the Z vectors to the inputs at each time step.
        if latent_z is not None:
            s = latent_z
        else:
            s = z
        r_loss, metric_map = self.decoder.reconstruction_loss(
            x_input, x_target, x_length, s, control_sequence
        )[0:2]

        free_nats = hparams.free_bits * tf.math.log(2.0)
        kl_cost = tf.maximum(kl_div - free_nats, 0)

        beta = ((1.0 - tf.pow(hparams.beta_rate, tf.to_float(self.global_step))) * hparams.max_beta)
        self.loss = tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost) + tf.reduce_mean(l2)

        scalars_to_summarize = {
            'loss': self.loss,
            'losses/r_loss': r_loss,
            'losses/kl_loss': kl_cost,
            'losses/kl_bits': kl_div / tf.math.log(2.0),
            'losses/kl_beta': beta,
            'losses/l2_loss': l2,
        }
        return metric_map, scalars_to_summarize

    def train(self, input_sequence, output_sequence, sequence_length, control_sequence=None):
        """
        Train on the given sequences, returning an optimizer.
        :param input_sequence: The sequence to be fed to the encoder.
        :param output_sequence: The sequence expected from the decoder.
        :param sequence_length: The length of the given sequences (which must be
            identical).
        :param control_sequence: (Optional) sequence on which to condition. This will be
            concatenated depthwise to the model inputs for both encoding and decoding.
        :return:
            optimizer: A tf.train.Optimizer.
        """

        _, scalars_to_summarize = self._compute_model_loss(
            input_sequence, output_sequence, sequence_length, control_sequence
        )

        hparams = self.hparams
        lr = ((hparams.learning_rate - hparams.min_learning_rate) *
              tf.pow(hparams.decay_rate, tf.to_float(self.global_step)) +
              hparams.min_learning_rate)

        tf.summary.scalar('learning_rate', lr)
        for n, t in scalars_to_summarize.items():
            tf.summary.scalar(n, tf.reduce_mean(t))

        optimizer = tf.train.AdamOptimizer(lr)

        return optimizer

    def eval(self, input_sequence, output_sequence, sequence_length, control_sequence=None):
        """
        Evaluate on the given sequences, returning metric update ops.
        :param input_sequence: The sequence to be fed to the encoder.
        :param output_sequence: The sequence expected from the decoder.
        :param sequence_length: The length of the given sequences (which must be
            identical).
        :param control_sequence: (Optional) sequence on which to condition the decoder.
        :return:
            metric_update_ops: tf.metrics update ops.
        """
        metric_map, scalars_to_summarize = self._compute_model_loss(
            input_sequence, output_sequence, sequence_length, control_sequence
        )

        for n, t in scalars_to_summarize.items():
            metric_map[n] = tf.metrics.mean(t)

        metrics_to_values, metrics_to_update = (
            contrib_metrics.aggregate_metric_map(metric_map)
        )

        for metric_name, metric_value in metrics_to_values.items():
            tf.summary.scalar(metric_name, metric_value)

        return list(metrics_to_update.values())

    def sample(self, n, max_length=None, latent_z=None, c_input=None, **kwargs):
        """Sample with on optional conditional embedding `z`."""
        if latent_z is not None and latent_z.shape[0].value != n:
            raise ValueError(
                '`z` must have a first dimension that equals `n` when given. '
                'Got: %d vs %d' % (latent_z.shape[0].value, n)
            )
        if self.hparams.encoded_z_size and latent_z is None:
            tf.logging.warning(
                'Sampling from conditional model without `z`. Using random `z`.'
            )
            normal_shape = [n, self.hparams.encoded_z_size]
            normal_dist = tfp.distributions.Normal(
                loc=tf.zeros(normal_shape), scale=tf.ones(normal_shape)
            )
            latent_z = normal_dist.sample()

        z = self._decode_latent(latent_z).sample()

        return self.decoder.sample(n, max_length, z, c_input, **kwargs)
