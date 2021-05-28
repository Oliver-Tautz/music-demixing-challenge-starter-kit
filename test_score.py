import numpy as np
from openunmix import data
import os
from csv import DictWriter
from collections import defaultdict
from tqdm import tqdm
import sys

from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write

import simpleaudio as sa

umx_estimates_filepath = 'data/umx_results/'
estimates_filepath = 'data/results/'
references_filepath = 'data/test/'
mixtures_filepath = '/data_ssd/music-demixing-challenge-starter-kit/data/test/'

songs = os.listdir(references_filepath)[0:3]


def get_files(filepath):
    instStrList = ['bass', 'drums', 'other', 'vocals']
    files = []
    for instStr in instStrList:
        files.append(data.load_audio(f"{filepath}/{instStr}.wav")[0].T.numpy())
    return np.array(files)



# replace arr1[ix] with arr2[ix]
def replace_with(arr1, arr2, ix):
    arr1 = arr1.copy()
    for i in ix:
        arr1[i] = arr2[i]
    return arr1


def get_inst_from_ix(ix):
    instStrList = ['bass', 'drums', 'other', 'vocals']

    l = []
    for i in ix:
        l.append(instStrList[i])
    return l


def write_sdr_values(references_d, estimates_d, filename):
    os.makedirs('scores_test', exist_ok=True)

    replacements = [[], [0], [1], [2], [3], [0, 1, 2, 3]]

    writedict = defaultdict(lambda: [])

    for song in tqdm(songs, f'computing values for {filename}'):
        reference = references_d[song]
        estimate = estimates_d[song]

        for replacement_ix in replacements:
            writedict['song'].append(song)
            writedict['replaced_with_GT'].append(get_inst_from_ix(replacement_ix))
            estimate_rep = replace_with(estimate, reference, replacement_ix)
            sdr_instr = sdr(reference, estimate_rep)
            sdr_song = np.mean(sdr_instr)
            writedict['sdr_song'].append(sdr_song)
            writedict['sdt_instr'].append(sdr_instr)

    csv_file = open(f'scores_test/{filename}', 'w', newline='')
    writer = DictWriter(csv_file, fieldnames=writedict.keys(), delimiter=';')
    writer.writeheader()

    for i in tqdm(range(len(writedict['song'])), desc=f'writing {filename}'):

        wd = dict()
        for key in writedict.keys():
            wd[key] = writedict[key][i]

        writer.writerow(wd)
    csv_file.close()

def write_sdr_values_lowpass(references_d, estimates_d, filename):
    os.makedirs('scores_test', exist_ok=True)


    writedict = defaultdict(lambda: [])


    for song in tqdm(songs, f'computing values for {filename}'):
        mixture_filename = f"{mixtures_filepath}{song}/mixture.wav"
        mixture = data.load_audio(mixture_filename)[0].T.numpy()
        write(f'filter_test/{song}_mixture.wav', 44100, mixture)
        for order in tqdm(range(1, 11),desc='testing_order'):
            #for cutoff in tqdm(range(100, 1000,50),desc='testing_fs'):
                cutoff = 450
                filtered = butter_lowpass_filter(mixture, cutoff, order=order)

                write(f'filter_test/{song}_{order}_filtered.wav', 44100, mixture)
                reference = references_d[song]
                estimate = estimates_d[song]


                # replace bass prediction with low pass filtered
                estimate_rep = replace_with(estimate, filtered, [0])

                sdr_instr = sdr(reference, estimate_rep)
                sdr_song_lpf = np.mean(sdr_instr)
                sdr_song = np.mean(sdr(reference,estimate))

                writedict['song'].append(song)
                writedict['butter_order'].append(order)
                writedict['butter_cutoffFs'] .append(cutoff)
                writedict['sdr_song'].append(sdr_song)
                writedict['sdr_song_lpf'].append(sdr_song_lpf)
                writedict['sdt_instr'].append(sdr_instr)

    csv_file = open(f'scores_test/{filename}', 'w', newline='')
    writer = DictWriter(csv_file, fieldnames=writedict.keys(), delimiter=';')
    writer.writeheader()

    for i in tqdm(range(len(writedict['song'])), desc=f'writing {filename}'):

        wd = dict()
        for key in writedict.keys():
            wd[key] = writedict[key][i]

        writer.writerow(wd)

    csv_file.close()

def butter_lowpass_filter(data, cutoff, fs=44100, order=3):
    data = data.copy().T

    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y0 = filtfilt(b, a, data[0])
    y1 = filtfilt(b, a, data[1])
    y = np.array([y0, y1]).T

    return y


def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)


# Get numpy arrays.

if len(sys.argv) > 1:

    references_dict = dict()
    estimates_dict = dict()
    umx_estimates_dict = dict()

    for song in tqdm(songs, desc='Loading Files'):
        references_dict[song] = get_files(f"{references_filepath}{song}/")
        estimates_dict[song] = get_files(f"{estimates_filepath}{song}/")
        umx_estimates_dict[song] = get_files(f"{umx_estimates_filepath}{song}/")

# sdr_instr = sdr(references, estimates)
# sdr_song = np.mean(sdr_instr)

# print(f'SDR for individual instruments: {sdr_instr}')
# print(f'SDR for full song: {sdr_song}')

# write_sdr_values(references_dict,umx_estimates_dict,'umx.csv')
# write_sdr_values(references_dict,estimates_dict,'scaled_mixture.csv')

write_sdr_values_lowpass(references_dict,umx_estimates_dict,'umx_vs_lowpass.csv')
write_sdr_values_lowpass(references_dict,estimates_dict,'scaled_mixture_vs_lowpass.csv')




