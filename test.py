#!/usr/bin/env python
#
# This file uses Demucs for music source speration, trained on Musdb-HQ
# See https://github.com/facebookresearch/demucs for more information
# The model was trained with the following flags (see the Demucs repo)
# python run.py --channels=48 --musdb=PATH_TO_MUSDB_HQ --is_wav
# **For more information, see**: https://github.com/facebookresearch/demucs/blob/master/docs/mdx.md
#
# NOTE: Demucs needs the model to be submitted along with your code.
# In order to download it, simply run once locally `python test.py`
#
# Making submission using the pretrained Demucs model:
# 1. Edit the `aicrowd.json` file to set your AICrowd username.
# 2. Submit your code using git-lfs
#    #> git lfs install
#    #> git add models
# 3. Download the pre-trained model by running
#    #> python test.py
#
# IMPORTANT: if you train your own model, you must follow a different procedure.
# When training is done in Demucs, the `demucs/models/` folder will contain
# the final trained model. Copy this model over to the `models/` folder
# in this repository, and add it to the repo (Make sure you setup git lfs!)
# Then, to load the model, see instructions in `prediction_setup()` hereafter.
import os

import torch.hub
import torch
import torchaudio as ta

from demucs import pretrained
from demucs.utils import apply_model, load_model  # noqa
from os.path import join
from subprocess import run
from evaluator.music_demixing import MusicDemixingPredictor
import denoiser.enhance

torch.set_num_threads(8)


class DemucsPredictor(MusicDemixingPredictor):

    def __init__(self, noise_reduction=False, noise_threshold=0.01):
        super().__init__()
        self.noise_reduction = noise_reduction
        self.noise_theshold = noise_threshold

    def prediction_setup(self):
        # Load your model here and put it into `evaluation` mode
        torch.hub.set_dir('./models/')


        print('loading_model')
        # Use a pre-trained model
        self.separator = pretrained.load_pretrained('demucs48_hq')

        # If you want to use your own trained model, copy the final model
        # from demucs/models (NOT demucs/checkpoints!!) into `models/` and then
        # uncomment the following line.

        # self.separator = load_model('models/my_model.th')

        self.separator.eval()

    def prediction(
            self,
            mixture_file_path,
            bass_file_path,
            drums_file_path,
            other_file_path,
            vocals_file_path,
    ):



        # Load mixture
        mix, sr = ta.load(str(mixture_file_path))
        assert sr == self.separator.samplerate
        assert mix.shape[0] == self.separator.audio_channels


        print(f'torch cuda avalable? {torch.cuda.is_available()}')
        print(f'torch cuda used? {next(self.separator.parameters()).is_cuda}')

        # Normalize track
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std

        # Separate
        with torch.no_grad():
            estimates = apply_model(self.separator, mix, shifts=5)
        estimates = estimates * std + mean

        # Store results
        target_file_map = {
            "vocals": vocals_file_path,
            "drums": drums_file_path,
            "bass": bass_file_path,
            "other": other_file_path,
        }
        for target, path in target_file_map.items():
            idx = self.separator.sources.index(target)
            source = estimates[idx]
            mx = source.abs().max()
            if mx >= 1:
                print('clipping', target, mx, std)
            source = source.clamp(-0.99, 0.99)

            if self.noise_reduction:
                source[source < self.noise_theshold] = 0
            # ta.save(str(path), source, sample_rate=sr)
            ta.save(str(path), source, sample_rate=sr, encoding='PCM_S', bits_per_sample=16)


# pretty crude implementation :/

class DemucsDoublePredictWrapper(DemucsPredictor):

    def __init__(self, tmp_dir='tmp',replace_instr=[]):
        super().__init__()
        self.tmp_dir = tmp_dir
        self.replace_instr =replace_instr
        os.makedirs(tmp_dir,exist_ok=True)

    def prediction(
            self,
            mixture_file_path,
            bass_file_path,
            drums_file_path,
            other_file_path,
            vocals_file_path,
    ):
        # predict one time.

        self.tmp_dir = os.path.dirname(bass_file_path)


        super().prediction(mixture_file_path, bass_file_path, drums_file_path, other_file_path,
                           vocals_file_path, )

        #gather input filenames
        inputs_files = []

        for instr in self.replace_instr:
            if instr == 'bass':
                inputs_files.append(bass_file_path)
            if instr == 'vocals':
                inputs_files.append(vocals_file_path)
            if instr == 'other':
                inputs_files.append(other_file_path)
            if instr == 'drums':
                inputs_files.append(drums_file_path)



        # predict second time for given instruments. (inst,mixturepath) = e.g. ('bass',bass_file_path)
        # use prediction as input for demucs.
        for inst, mixture_path in zip(self.replace_instr,inputs_files):
            super().prediction(mixture_path, join(self.tmp_dir, f'{inst}_bass.wav'), join(self.tmp_dir, f'{inst}_drums.wav'),
                               join(self.tmp_dir, f'{inst}_other.wav'),
                               join(self.tmp_dir, f'{inst}_vocals.wav'), )


        for instr in self.replace_instr:

            if instr == 'bass':
                print(run(['cp',join(self.tmp_dir,f"{instr}_{instr}.wav"),bass_file_path]))
            if instr == 'vocals':
                print(run(['cp',join(self.tmp_dir,f"{instr}_{instr}.wav"),vocals_file_path]))
            if instr == 'other':
                print(run(['cp',join(self.tmp_dir,f"{instr}_{instr}.wav"),other_file_path]))
            if instr == 'drums':
                print(run(['cp',join(self.tmp_dir,f"{instr}_{instr}.wav"),drums_file_path]))

       # run(['rm','-rf',self.tmp_dir])

class DemucsDenoiseWrapper(DemucsPredictor):

    def __init__(self, tmp_dir='tmp', denosie_instr=['vocals']):
        super().__init__()
        self.tmp_dir = tmp_dir
        self.replace_instr =denosie_instr
        os.makedirs(tmp_dir,exist_ok=True)

    def prediction(
            self,
            mixture_file_path,
            bass_file_path,
            drums_file_path,
            other_file_path,
            vocals_file_path,
    ):
        # predict one time.

        self.tmp_dir = os.path.dirname(bass_file_path)


        super().prediction(mixture_file_path, bass_file_path, drums_file_path, other_file_path,
                           vocals_file_path, )

        #gather input filenames
        inputs_files = []

        for instr in self.replace_instr:
            if instr == 'bass':
                inputs_files.append(bass_file_path)
            if instr == 'vocals':
                inputs_files.append(vocals_file_path)
            if instr == 'other':
                inputs_files.append(other_file_path)
            if instr == 'drums':
                inputs_files.append(drums_file_path)



        # predict second time for given instruments. (inst,mixturepath) = e.g. ('bass',bass_file_path)
        # use prediction as input for demucs.
        for inst, mixture_path in zip(self.replace_instr,inputs_files):
            super().prediction(mixture_path, join(self.tmp_dir, f'{inst}_bass.wav'), join(self.tmp_dir, f'{inst}_drums.wav'),
                               join(self.tmp_dir, f'{inst}_other.wav'),
                               join(self.tmp_dir, f'{inst}_vocals.wav'), )


        for instr in self.replace_instr:

            if instr == 'bass':
                run(['cp',join(self.tmp_dir,f"{instr}_{instr}.wav"),bass_file_path])
            if instr == 'vocals':
                print(run(['cp',join(self.tmp_dir,f"{instr}_{instr}.wav"),vocals_file_path]))
            if instr == 'other':
                run(['cp',join(self.tmp_dir,f"{instr}_{instr}.wav"),other_file_path])
            if instr == 'drums':
                run(['cp',join(self.tmp_dir,f"{instr}_{instr}.wav"),drums_file_path])



if __name__ == "__main__":
    submission = DemucsDoublePredictWrapper()
    submission.run()
    print("Successfully generated predictions!")
