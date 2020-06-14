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

"""A class for loading trained MusicVAE models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tarfile
import numpy as np

from backports import tempfile
import tensorflow.compat.v1 as tf   # pylint: disable=import-error
from magenta.common import merge_hparams
from configs import Config


def _update_config(config1, config2):
    """Update config1 hparams with hparams from config2."""
    h = merge_hparams(config1.hparams, config2.hparams)
    return Config(
        model=config1.model,
        hparams=h,
        note_sequence_augmenter=config1.note_sequence_augmenter,
        data_converter=config1.data_converter,
        train_examples_path=config1.train_examples_path,
        eval_examples_path=config1.eval_examples_path
    )


class TrainedModel(object):
    """An interface to a trained model for encoding, decoding, and sampling.

    Args:
        vae_config: The Config to update hparams for model config.
        model_config: The Config to build the model graph with.
        batch_size: The batch size to build the model graph with.
        vae_checkpoint_dir_or_path: The directory containing VAE checkpoints for the model,
            the most recent of which will be loaded, or a direct path to a specific
            checkpoint.
        model_checkpoint_dir_or_path: The directory containing our
         training checkpoints for the model, the most recent of which will be loaded,
          or a direct path to a specific checkpoint.
        model_var_pattern: List of string containing variable name patterns
         we want to restore from our model.
        session_target: Optional execution engine to connect to. Defaults to
            in-process.
        sample_kwargs: Additional, non-tensor keyword arguments to
        pass to sample call.
    """

    def __init__(
            self, vae_config, model_config, batch_size, vae_checkpoint_dir_or_path=None,
            model_checkpoint_dir_or_path=None, model_var_pattern=None,
            session_target='', **sample_kwargs):
        if tf.gfile.IsDirectory(vae_checkpoint_dir_or_path):
            vae_checkpoint_path = tf.train.latest_checkpoint(vae_checkpoint_dir_or_path)
        else:
            vae_checkpoint_path = vae_checkpoint_dir_or_path

        if tf.gfile.IsDirectory(model_checkpoint_dir_or_path):
            model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir_or_path)
        else:
            model_checkpoint_path = model_checkpoint_dir_or_path
        self._config = _update_config(model_config, vae_config)
        self._config.data_converter.set_mode('infer')
        self._config.hparams.batch_size = batch_size
        with tf.Graph().as_default():
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
                self._latent_z_input = tf.placeholder(
                    tf.float32, shape=[batch_size, self._config.hparams.encoded_z_size])
            else:
                self._latent_z_input = None

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
                latent_z=self._latent_z_input,
                c_input=self._c_input,
                temperature=self._temperature,
                **sample_kwargs)

            vae_var_list = []
            model_var_list = []
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                flag = False
                for pattern in model_var_pattern:
                    if re.search(pattern, v.name):
                        flag = True
                        model_var_list.append(v)
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

            # Restore model graph part
            model_saver = tf.train.Saver(model_var_list)
            if os.path.exists(vae_checkpoint_path) and tarfile.is_tarfile(model_checkpoint_path):
                tf.logging.info('Unbundling model checkpoint.')
                with tempfile.TemporaryDirectory() as temp_dir:
                    tar = tarfile.open(vae_checkpoint_path)
                    tar.extractall(temp_dir)
                    # Assume only a single checkpoint is in the directory.
                    for name in tar.getnames():
                        if name.endswith('.index'):
                            model_checkpoint_path = os.path.join(temp_dir, name[0:-6])
                            break
                    model_saver.restore(self._sess, model_checkpoint_path)
            else:
                model_saver.restore(self._sess, model_checkpoint_path)

    def sample(self, n=None, length=None, temperature=1.0, same_latent_z=False, c_input=None):
        """
        Generates random samples from the model.
        :param n: The number of samples to return. A full batch will be returned if not specified.
        :param length: The maximum length of sample in decoder iterations. Required
            if end tokens are not being used.
        :param temperature: The softmax temperature to use (if applicable)
        :param same_latent_z: Whether to use the same latent vector
        for all samples in the batch (if applicable).
        :param c_input: A sequence of control inputs to use for all samples (if applicable).
        :return:
            A list of samples as NoteSequence objects.
        :raises:
            ValueError: If `length` is not specified and an end token is not being used.
        """
        batch_size = self._config.hparams.batch_size
        n = n or batch_size
        latent_z_size = self._config.hparams.encoded_z_size

        if not length and self._config.data_converter.end_token is None:
            raise ValueError(
                'A length must be specified when the end token is not used.'
            )
        length = length or tf.int32.max

        feed_dict = {
            self._temperature: temperature,
            self._max_length: length
        }

        if self._latent_z_input is not None and same_latent_z:
            latent_z = np.random.randn(latent_z_size).astype(np.float32)
            latent_z = np.tile(latent_z, (batch_size, 1))
            feed_dict[self._latent_z_input] = latent_z

        if self._c_input is not None:
            feed_dict[self._c_input] = c_input

        outputs = []
        for _ in range(int(np.ceil(n / batch_size))):
            if self._latent_z_input is not None and not same_latent_z:
                feed_dict[self._latent_z_input] = (
                    np.random.randn(batch_size, latent_z_size).astype(np.float32)
                )
            outputs.append(self._sess.run(self._outputs, feed_dict))
        samples = np.vstack(outputs)[:n]
        if self._c_input is not None:
            return self._config.data_converter.from_tensors(
                samples, np.tile(np.expand_dims(c_input, 0), [batch_size, 1, 1])
            )

        return self._config.data_converter.from_tensors(samples)
