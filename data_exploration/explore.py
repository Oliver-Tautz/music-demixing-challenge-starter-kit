import soundfile as sf
import os
import plotly as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from simple_logger import SimpleLogger
from multiprocessing import Process, Lock
import librosa

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--plot', action='store_true',help='really plot. Overwrites existing images!')
parser.add_argument('--picture-format',  default='jpg', type=str ,help='really plot. Overwrites existing images!')
args = parser.parse_args()



exploration_folder = "/home/olli/gits/music_demixing_challenge_gits/demucs/data_exploration/"
prediction_folder = "/home/olli/gits/music_demixing_challenge_gits/demucs/data/results"
reference_folder = "/home/olli/gits/music_demixing_challenge_gits/demucs/data/test"

metrics_folder = os.path.join(exploration_folder,'metrics')
plots_folder = os.path.join(exploration_folder,'plots')

features = ['zero_crossing_rate','spectral_centroid','spectral_rolloff']

ref_features = ['ref_'+ x for x in features]
pred_features = ['pred_'+ x for x in features]

os.makedirs(metrics_folder,exist_ok=True)
os.makedirs(plots_folder,exist_ok=True)

LEFT = 0
RIGHT = 1



chunk_seconds = 0.5

# not used
max_threads = 8

instruments = ['drums','bass','vocals','other']
plot_filetype = args.picture_format
dry_run = not args.plot


def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=0)
    den = np.sum(np.square(references - estimates), axis=0)
    num += delta
    den += delta
    return 10 * np.log10(num / den)


def sdr_nolog(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=0)
    den = np.sum(np.square(references - estimates), axis=0)
    num += delta
    den += delta
    return (num / den)



def visual_loss(reference,prediction):
    return np.square(reference-prediction)

def get_features(wav,sr):
    zero_crossing = librosa.zero_crossings(wav).sum()
    spectral_centroid = librosa.feature.spectral_centroid(wav,sr=sr,)[0].mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(wav,sr=sr)[0].mean()


    return zero_crossing,spectral_centroid,spectral_rolloff

def plot_loss_graph_plt(songame,instr,chunkid,reference, prediction,loss, loss_max,loss_min,wav_max,wav_min,filename=None, show=False,noplot=True):
    alpha = 0.2
    assert len(reference) == len(prediction)

    chunkloss_nolog = sdr_nolog(reference, prediction)
    chunkloss = sdr(reference, prediction)


    # plot on right y-axis
    if not dry_run:
        fig, (ax2, ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)
        # ax2 = ax.twinx()

        time = np.linspace(0, len(reference), len(reference))

        ax2.plot(time, reference, alpha=alpha, label='reference')
        ax2.plot(time, prediction, alpha=alpha,color='green', label='prediction')
        ax.plot(time, loss, label='loss', color='red',alpha=0.2)

        ax.set_ylabel("se_loss")

        ax.set_ylim(loss_min, loss_max)

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc='upper left', prop={'size': 6})
        fig.suptitle(f"{songame}, {instr}, id={chunkid}\nsdr = {chunkloss}\n mse= {loss.mean()}")

        ax2.set_ylabel("wav")
        ax2.set_ylim(wav_min,wav_max)


    if not show:
        if filename:
            if not dry_run:
                plt.savefig(filename,pil_kwargs={'quality': 60},dpi=150)
                plt.close(fig)
        else:
            print("no filename supplied, ignoring!")
    else:
        plt.show()
        plt.close(fig)




    return chunkloss, chunkloss_nolog


def wait_threads(threads):
    for thread in threads:
        thread.join()


def plot_song(songname):
    csv_file = os.path.join(metrics_folder, f"{songname}.csv")
    colnames = ['song', 'inst', 'model', 'chunklen_s', 'channel', 'second', 'chunk',
     'sdr', 'sdr_nolog', 'mse_loss']

    global  ref_features
    global pred_features

    colnames.extend(ref_features)
    colnames.extend(pred_features)

    logger = SimpleLogger(filename=csv_file,
                          colnames=colnames)
    for instrument in instruments:


        logger_lock = Lock()

        ref_path = os.path.join(reference_folder, songname, f"{instrument}.wav")
        pred_path = os.path.join(prediction_folder, songname, f"{instrument}.wav")

        ref, ref_fs = sf.read(ref_path)
        pred, pred_fs = sf.read(pred_path)

        assert ref_fs == pred_fs
        assert ref.shape == pred.shape

        # half a second per plot

        chunklen = ref_fs * chunk_seconds

        for channel in [LEFT, RIGHT]:
            ref_c = ref[:, channel]
            pred_c = pred[:, channel]
            loss = visual_loss(ref_c, pred_c)

            loss_max = loss.max()
            loss_min = loss.min()

            wav_max = max(ref_c.max(), pred_c.max())
            wav_min = min(ref_c.min(), pred_c.min())

            ref_chunks = np.array_split(ref_c, len(ref_c) // chunklen)
            pred_chunks = np.array_split(pred_c, len(ref_c) // chunklen)
            loss_chunks = np.array_split(loss, len(ref_c) // chunklen)

            for i, (ref_chunk, pred_chunk, loss_chunk) in enumerate(
                    tqdm(zip(ref_chunks, pred_chunks, loss_chunks), total=len(ref_chunks))):

                ref_features = get_features(ref_chunk,pred_fs)
                pred_features = get_features(pred_chunk,ref_fs)
                filename = os.path.join(plots_folder, songname,
                                        f'{instrument}_{channel}_{i}.{plot_filetype}')
                mse_loss = loss.mean()
                loss, loss_nolog = plot_loss_graph_plt(songname, instrument, i, ref_chunk, pred_chunk,
                                                       loss_chunk, loss_max, loss_min, wav_max, wav_min,
                                                       filename=filename, show=False)

                logger.log([songname, instrument, 'demucs', chunk_seconds, channel, i * chunk_seconds, i, loss,
                            loss_nolog,mse_loss,*ref_features,*pred_features])


if __name__ == "__main__":

    #for songname in os.listdir(prediction_folder):
    # thread by song

    threads = []
    
    for songname in os.listdir(prediction_folder):
        os.makedirs(os.path.join(plots_folder,songname),exist_ok=True)
        p= Process(target = plot_song, args = [songname])
        p.start()
        threads.append(p)

        
    wait_threads(threads)




# not required to thread this
#       processes = []
#       for ref_chunk, pred_chunk in tqdm(zip(ref_chunks,pred_chunks),total=len(ref_chunks)):
#
#           while len(processes) < max_threads:
#               p = Process(target=plot_loss_graph_plt,args=[ref_chunk,pred_chunk,False])
#               processes.append(p)
#               p.start()
#           wait_threads(processes)
#
#       wait_threads(processes)
