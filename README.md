# LCMusicVAE: A latent constraint model for music variational autoencoder.

LCMusicVAE learns a smaller latent space from MusicVAE to encapsulate the general characteristic of a song. Thus, it can be trained
faster and more energy efficient. Under the hood, this is just a MidiMe [model](https://magenta.tensorflow.org/midi-me) from Google
open source [Magenta](https://github.com/tensorflow/magenta) but implemented in Python. For more detail behind the model, please take
a look at this [paper](https://research.google/pubs/pub48628/).

## Installation:

Currently I do not support python3. Preferably using Anaconda to create new environment by:

```
conda create -n magenta python=2.7
```

And pip install the following dependency:
* tensorflow==1.14.0
* tensorflow_datasets==1.3.2
* tensorflow_probability==0.7.0
* magenta==1.1.8

### How To Use

Download `cat-mel_2bar_big`	pre-trained [checkpoint](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar)
(Currently only support one model). There are two operations you can choose: `sample` and `interpolate`.

#### Sample

The mechanism is just similar to [MusicVAE](https://github.com/tensorflow/magenta/tree/master/magenta/models/music_vae) but we
also need to load MusicVAE model instead of only loading LCMusicVAE model.

```sh
python midime_generate.py \
--vae_config=cat-mel_2bar_big \
--config=lc-cat-mel_2bar_big \
--checkpoint_file=/path/to/lc_music_vae/checkpoints/lc-cat-mel_2bar_big.tar \
--vae_checkpoint_file=/path/to/music_vae/checkpoints/cat-mel_2bar_big.tar \
--mode=sample \
--num_outputs=5 \
--output_dir=/tmp/lc_music_vae/generated
```

#### Interpolate

Same thing applies here:
```sh
python midime_generate.py \
--config=lc-cat-mel_2bar_big \
--vae_config=cat-mel_2bar_big \
--checkpoint_file=/path/to/lc_music_vae/checkpoints/lc-cat-mel_2bar.ckpt \
--vae_checkpoint_file=/path/to/music_vae/checkpoints/cat-mel_2bar.ckpt \
--mode=interpolate \
--num_outputs=5 \
--input_midi_1=/path/to/input/1.mid
--input_midi_2=/path/to/input/2.mid
--output_dir=/tmp/music_vae/generated
```

#### Training your own LCMusicVAE

Just like MusicVAE training pipeline, we also need to convert a collection of MIDI files into NoteSequences by following the instructions
in [Building your Dataset](https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md). Addtionally, we
also need to load encoder part and latent space from MusicVAE pre-trained model for training smaller latent space. 
Assuming your generated examples are `data/notesequences.tfrecord`, training `lc-cat-mel_2bar_big` will be something like this:

```sh
python2 midime_train.py \
--config=lc-cat-mel_2bar_big \
--run_dir=tmp/ \
--mode=train \
--examples_path=data/notesequences.tfrecord \
--eval_num_batches=2 \
--pretrained_path=model/cat-mel_2bar_big.ckpt \
--num_steps=10
```
