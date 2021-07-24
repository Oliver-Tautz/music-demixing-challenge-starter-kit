from simple_logger import SimpleLogger
import numpy as np
from os.path import join
import soundfile as sf
from collections import defaultdict


tmp_dir = '/home/olli/gits/music_demixing_challenge_gits/demucs/tmp'
predict_dir = '/home/olli/gits/music_demixing_challenge_gits/demucs/data/results/'
reference_dir = '/home/olli/gits/music_demixing_challenge_gits/demucs/data/test/'


def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)

def sdr_dict(references,estimates):

    song_score = sdr(references,estimates)

    scores = dict()

    scores["bass"] = song_score[0]
    scores["drums"] = song_score[1]
    scores["other"] = song_score[2]
    scores["vocals"] = song_score[3]
    scores["total"] = np.mean(song_score)

    return scores

logger = SimpleLogger('./double_predict.csv',['type','total_sdr','bass_sdr','drums_sdr','other_sdr','vocals_sdr'])



references = []
estimates = []

songname = 'Motor Tapes - Shore'
instruments = ["bass", "drums", "other", "vocals"]
double_predictions = defaultdict(lambda: dict())
for instrument in instruments:
    reference_file = join(reference_dir, songname, instrument + ".wav")
    estimate_file = join(predict_dir,songname,instrument+".wav")


    for instrument2 in instruments:
        double_predict_filepath = join(tmp_dir,f"{instrument}_{instrument2}.wav")
        double_predictions[instrument][instrument2] = sf.read(double_predict_filepath)[0]

    reference, _ = sf.read(reference_file)
    estimate, _ = sf.read(estimate_file)
    references.append(reference)
    estimates.append(estimate)

references = np.stack(references)
estimates = np.stack(estimates)

demucs_scores = sdr_dict(references,estimates)

print('demucs',demucs_scores)

estimates_single = []
estimates_sum = defaultdict(lambda : [])

for instrument in instruments:
    estimates_single.append(double_predictions[instrument][instrument])
    for instrument2 in instruments:
        estimates_sum[instrument].append(double_predictions[instrument][instrument2])
    estimates_sum[instrument] = sum(estimates_sum[instrument])

estimates_single = np.stack(estimates_single)
double_single_scores = sdr_dict(references,estimates_single)

print('double_single',double_single_scores)



estimates_sum = np.stack(list(estimates_sum.values()))
double_sum_scores = sdr_dict(references,estimates_sum)

print('double_sum',double_sum_scores)



