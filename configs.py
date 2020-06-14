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

"""Configurations for MusicVAE models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from magenta.common import merge_hparams
from magenta.models.music_vae import data
from magenta.models.music_vae import lstm_models
from tensorflow.contrib.training import HParams # pylint: disable=import-error

from base_model import LCMusicVAE, SmallMusicVAE


class Config(collections.namedtuple(
        'Config', [
            'model', 'hparams', 'note_sequence_augmenter', 'data_converter',
            'train_examples_path', 'eval_examples_path', 'tfds_name', 'pretrained_path',
            'var_train_pattern', 'encoder_train', 'decoder_train'])):
    """Config class."""
    def values(self):
        """Return value as dictionary."""
        return self._asdict()


Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
    """Update config with new values."""
    config_dict = config.values()
    config_dict.update(update_dict)
    return Config(**config_dict)


CONFIG_MAP = dict()

CONFIG_MAP['lc-cat-mel_2bar_big'] = Config(
    model=LCMusicVAE(lstm_models.BidirectionalLstmEncoder(),
                     lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=2,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            encoded_z_size=8,
            enc_rnn_size=[2048],
            dec_rnn_size=[128, 128],
            free_bits=0,
            max_beta=0.5,
            beta_rate=0.99999,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
    pretrained_path=None,
    var_train_pattern=['latent_encoder', 'decoder'],
    encoder_train=False,
    decoder_train=True
)

CONFIG_MAP['ae-cat-mel_2bar_big'] = Config(
    model=SmallMusicVAE(lstm_models.BidirectionalLstmEncoder(),
                        lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=2,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            encoded_z_size=4,
            latent_encoder_layers=[1024, 256, 64],
            latent_decoder_layers=[64, 256, 1024],
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
            max_beta=0.5,
            beta_rate=0.99999,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
    pretrained_path=None,
    var_train_pattern=['latent'],
    encoder_train=False,
    decoder_train=False
)
