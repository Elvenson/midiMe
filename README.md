# SmallMusicVAE: An encoded latent space model for music variational autoencoder.

`SmallMusicVAE` is simply a [MidiMe model](https://magenta.tensorflow.org/midi-me) from Google open source
[Magenta](https://github.com/tensorflow/magenta) but implemented in Python instead of JavaScript from original model.
This model learns a smaller latent space from MusicVAE to encapsulate the general characteristic of a song. Thus, it can
be trained faster and more energy efficient. For more detail behind the model, please take a look at this [paper](https://research.google/pubs/pub48628/).

Additionally, this repo also includes `LCMusicVAE` which is a latent constraint MusicVAE model. Due to its training structure,
each pre-trained model can be viewed as a module, making it easier to achieve latent space transfer between domains. 
This implementation is based from this [paper](https://arxiv.org/pdf/1902.08261.pdf) but only on one domain music (For experimentation purpose).

Here is my [medium post](https://medium.com/@bobi_29852/smallmusicvae-an-encoded-latent-space-model-for-music-variational-autoencoder-e087c7fd2536)
to briefly explain the architecture of both models as well as some melody samples generated from `SmallMusicVAE` model you can listen to.

## Installation:

Just like official [Magenta](https://github.com/tensorflow/magenta) repo, you can `pip install magenta==1.3.1` package 
(tested only on Python == 3.7) or if you want to install with anaconda, just simply type:

```bash
curl https://raw.githubusercontent.com/tensorflow/magenta/master/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
bash /tmp/magenta-install.sh
```

Or you can simply create new python3.7 environment, change directory to this repo and type the following line:
```bash
pip install -r requirement.txt
```
Do note that it's my working environment so there might be some unused packages.


## How To Use

Get `cat-mel_2bar_big` pre-trained musicVAE model (download [here](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar))

### Small Autoencoder MusicVAE (SmallMusicVAE):

#### Training
Just like MusicVAE training pipeline, we also need to convert a collection of MIDI files into NoteSequences by following the instructions
in [Building your Dataset](https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md). Additionally, we
also need to load encoder, decoder part and latent space from MusicVAE pre-trained model for training smaller latent space. 
Assuming your generated examples are `data/notesequences.tfrecord`, training `ae-cat-mel_2bar_big` will be something like this:

```sh
python midime_train.py \
--config=ae-cat-mel_2bar_big \
--run_dir=<model output directory> \
--mode=train \
--examples_path=<path to sample tfrecord> \
--pretrained_path=<path to cat-mel_2bar_big.ckpt> \
--num_steps=100
```

#### Sample
After training we can load the model for melody generation. The mechanism is just similar to [MusicVAE](https://github.com/tensorflow/magenta/tree/master/magenta/models/music_vae)
but we also need to load MusicVAE model instead of only loading SmallMusicVAE model.

```sh
python midime_generate.py \
--vae_config=cat-mel_2bar_big \
--config=ae-cat-mel_2bar_big \
--checkpoint_file=<path to your train model ckpt file> \
--vae_checkpoint_file=<path to cat-mel_2bar_big.ckpt> \
--num_outputs=5 \
--output_dir=<generated melody output path>
```

### Latent Constraint MusicVAE (LCMusicVAE):

#### Training
Similar to `SmallMusicVAE`, we just need to change the `config` value to `lc-cat-mel_2bar_big`:
```sh
python midime_train.py \
--config=lc-cat-mel_2bar_big \
--run_dir=<model output directory> \
--mode=train \
--examples_path=<sample tfrecord file> \
--pretrained_path=<path to cat-mel_2bar_big.ckpt> \
--num_steps=100
```

#### Sample
Same thing applies here, we just need to change `config` parameter to `lc-cat-mel_2bar_big`:

```sh
python midime_generate.py \
--vae_config=cat-mel_2bar_big \
--config=lc-cat-mel_2bar_big \
--checkpoint_file=<path to your model ckpt file> \
--vae_checkpoint_file=<path to cat-mel_2bar_big.ckpt> \
--num_outputs=5 \
--output_dir=<generated melody output path>
```

### Example
For ease of testing, I have put some generated sample data in `data` folder. 
To test the training phase, you can try this command:
```shell script
python midime_train.py \
--config=ae-cat-mel_2bar_big \
--run_dir=tmp/ \
--mode=train \
--examples_path=data/fur_elise.tfrecord \
--pretrained_path=<path to cat-mel_2bar_big.ckpt> \
--num_steps=200
```
Your model will be in `tmp/train` folder. 

For generating melody, you can try:
```shell script
python midime_generate.py \
--vae_config=cat-mel_2bar_big \
--config=ae-cat-mel_2bar_big \
--checkpoint_file=tmp/train/model.ckpt-200 \
--vae_checkpoint_file=<path to cat-mel_2bar_big.ckpt> \
--num_outputs=15 \
--output_dir=tmp/generated
```
Your generated melodies will be in `tmp/generated` folder.