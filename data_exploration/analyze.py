import argparse
import os

from explore import metrics_folder, plots_folder, ref_features, pred_features
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from sklearn.cluster import KMeans
from tqdm import tqdm
from simple_logger import SimpleLogger
from collections import Counter

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--plot', action='store_true', help='really plot. Overwrites existing images!')
parser.add_argument('--picture-format', default='jpg', type=str, help='really plot. Overwrites existing images!')

args = parser.parse_args()

csv_seperator = ';'

metrics_plot_folder = os.path.join(metrics_folder, f"metrics_plots")
cluster_folder = os.path.join(metrics_folder, f"clusters")
os.makedirs(metrics_plot_folder, exist_ok=True)
os.makedirs(cluster_folder, exist_ok=True)

no_clusters = 10




def kmeans_clustering(df, column, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit((df[column].to_numpy()))
    clusters = kmeans.predict(df[column].to_numpy())
    return clusters, kmeans.inertia_


if __name__ == "__main__":

    log_cols = ['inst', 'cluster', 'sdr_mean', 'sdr_std', 'sdr_nolog_mean', 'sdr_nolog_std', 'size', 'mse_mean',
                'mse_std']
    log_cols.extend([x + '_mean' for x in ref_features])
    log_cols.extend([x + '_std' for x in ref_features])
    cluster_logger = SimpleLogger(os.path.join(cluster_folder, "cluster.csv"), log_cols)
    dataframes = []
    for song_csv_filename in os.listdir(metrics_folder):
        full_path = os.path.join(metrics_folder, song_csv_filename)
        if os.path.isdir(full_path):
            continue
        df = pd.read_csv(full_path, sep=csv_seperator)
        dataframes.append(df)
    df = pd.concat(dataframes)

    for inst in tqdm(['bass', 'drums', 'other', 'vocals']):

        n = df.shape[0]
        df_inst = df.loc[df['inst'] == inst]

        axl = df_inst.hist(column='sdr', bins=int(np.sqrt(n)), grid=False)
        axl[0][0].set_xlim(-80, 50)
        plt.suptitle(f"{inst}")
        plt.savefig(os.path.join(metrics_plot_folder, f"{inst}_sdr_hist.jpg"), pil_kwargs={'quality': 70})
        plt.clf()

        df_inst_clean = df.loc[(df['inst'] == inst) & (df['sdr'] > -1)]
        df_inst_clean.hist(column='sdr', bins=int(np.sqrt(n)), grid=False)
        plt.savefig(os.path.join(metrics_plot_folder, f"{inst}_sdr_hist_positive.jpg"), pil_kwargs={'quality': 70})
        plt.clf()

        clustering, _ = kmeans_clustering(df_inst, ['sdr', 'ref_spectral_centroid', 'ref_spectral_rolloff',
                                                    'ref_zero_crossing_rate'], no_clusters)

        df_inst.insert(len(df.columns), 'cluster', clustering)

        #   inertias = []
        #   for k in range(1, 30):
        #       print(k)
        #       clustering, inertia = kmeans_clustering(df, ['sdr','ref_spectral_centroid','ref_spectral_rolloff','ref_zero_crossing_rate'], k)
        #       inertias.append((k, inertia))
        #
        #   plt.plot(inertias)
        #   plt.show()

        for cluster in np.unique(clustering):
            print()
            df_inst_cluster = df_inst.loc[df_inst['cluster'] == cluster]

            sdr_mean = df_inst_cluster['sdr'].mean()
            sdr_std = df_inst_cluster['sdr'].std()
            mse_mean = df_inst_cluster['mse_loss'].mean()
            mse_std = df_inst_cluster['mse_loss'].std()
            sdr_nolog_mean = df_inst_cluster['sdr_nolog'].mean()
            sdr_nolog_std = df_inst_cluster['sdr_nolog'].std()

            ref_spectral_centroid_mean = df_inst_cluster['ref_spectral_centroid'].mean()
            ref_spectral_rolloff_mean = df_inst_cluster['ref_spectral_rolloff'].mean()
            ref_zero_crossing_rate_mean = df_inst_cluster['ref_zero_crossing_rate'].mean()

            ref_spectral_centroid_std = df_inst_cluster['ref_spectral_centroid'].std()
            ref_spectral_rolloff_std = df_inst_cluster['ref_spectral_rolloff'].std()
            ref_zero_crossing_rate_std = df_inst_cluster['ref_zero_crossing_rate'].std()

            size = df_inst_cluster.shape[0]

            cluster_logger.log(
                [inst, cluster, sdr_mean, sdr_std, sdr_nolog_mean, sdr_nolog_std, size, mse_mean, mse_std,
                 ref_zero_crossing_rate_mean, ref_spectral_centroid_mean, ref_spectral_rolloff_mean,
                 ref_zero_crossing_rate_std, ref_spectral_centroid_std, ref_spectral_rolloff_std])
            os.makedirs(os.path.join(cluster_folder, inst, str(cluster)), exist_ok=True)

        clustercounter = Counter()
        for _, line in tqdm(df_inst.iterrows(),total=df_inst.shape[0]):
            songname = line['song']
            inst = line['inst']
            channel = line['channel']
            chunk_id = line['chunk']
            cluster = line['cluster']
            filename = os.path.join(plots_folder, songname, f"{inst}_{channel}_{chunk_id}.jpg")
            clustercounter[cluster]+=1

            if not (os.path.isfile(filename)):
                print("file_not_found!")
            else:
                if args.plot :
                    p = subprocess.run(["cp", filename, os.path.join(cluster_folder, inst, str(cluster),
                                                                 f"{songname}_{channel}_{chunk_id}.jpg")], stdin=None,
                                   stdout=None, stderr=None, close_fds=True)
                    if p.returncode != 0:
                        print(p.returncode)

        print(inst,clustercounter)

    # inertias = []
    # for k in range(1,30):
    #     print(k)
    #     clustering, inertia = kmeans_clustering(df, ['sdr', 'sdr_nolog'], k)
    #     inertias.append((k,inertia))
    #
    # plt.plot(inertias)
    # plt.show()

    axl = df_inst.hist(column='sdr', bins=int(np.sqrt(n)), grid=False)
    axl[0][0].set_xlim(-80, 50)
    plt.suptitle(f"all")
    plt.savefig(os.path.join(metrics_plot_folder, f"all_sdr_hist.jpg"), pil_kwargs={'quality': 70})
    plt.clf()

    clustering, _ = kmeans_clustering(df, ['sdr', 'ref_spectral_centroid', 'ref_spectral_rolloff',
                                           'ref_zero_crossing_rate'], no_clusters)
    df.insert(len(df.columns), 'cluster', clustering)

    for cluster in np.unique(clustering):
        df_cluster = df.loc[df['cluster'] == cluster]

        sdr_mean = df_cluster['sdr'].mean()
        sdr_std = df_cluster['sdr'].std()
        sdr_nolog_mean = df_cluster['sdr_nolog'].mean()
        sdr_nolog_std = df_cluster['sdr_nolog'].std()
        mse_mean = df_cluster['mse_loss'].mean()
        mse_std = df_cluster['mse_loss'].std()

        ref_spectral_centroid_mean = df_cluster['ref_spectral_centroid'].mean()
        ref_spectral_rolloff_mean = df_cluster['ref_spectral_rolloff'].mean()
        ref_zero_crossing_rate_mean = df_cluster['ref_zero_crossing_rate'].mean()

        ref_spectral_centroid_std = df_cluster['ref_spectral_centroid'].std()
        ref_spectral_rolloff_std = df_cluster['ref_spectral_rolloff'].std()
        ref_zero_crossing_rate_std = df_cluster['ref_zero_crossing_rate'].std()

        size = df_cluster.shape[0]

        cluster_logger.log(['all', cluster, sdr_mean, sdr_std, sdr_nolog_mean, sdr_nolog_std, size, mse_mean, mse_std,
                 ref_zero_crossing_rate_mean, ref_spectral_centroid_mean, ref_spectral_rolloff_mean,
                 ref_zero_crossing_rate_std, ref_spectral_centroid_std, ref_spectral_rolloff_std])

        os.makedirs(os.path.join(cluster_folder, 'all', str(cluster)), exist_ok=True)

    for _, line in tqdm(df.iterrows()):
        songname = line['song']
        inst = line['inst']
        channel = line['channel']
        chunk_id = line['chunk']
        cluster = line['cluster']
        filename = os.path.join(plots_folder, songname, f"{inst}_{channel}_{chunk_id}.jpg")

        if not (os.path.isfile(filename)):
            print("file_not_found!")
        else:
            if args.plot:
                subprocess.run(["cp", filename, os.path.join(cluster_folder, "all", str(cluster),
                                                             f"{inst}_{songname}_{channel}_{chunk_id}.jpg")], stdin=None,
                               stdout=None, stderr=None, close_fds=True)
