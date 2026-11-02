import numpy as np
import os
from collections import Counter
import cv2
import math
from track_cluster import get_traj_cluster,get_blazeit_labels, get_m30_labels 

class cluster_traj(object):

    def __init__(self,video_name,background_img,origin_tuples,min_traj_num,thresh_num=100):

        self.video_name = video_name
        self.track_filtered = []
        self.origin_tracks = []
        self.track_clustered = []
        self.background = background_img
        self.origin_tuples = origin_tuples
        self.min_traj_num = min_traj_num
        self.thresh_num = thresh_num

    def filter_traj(self):

        tracks = []
        for key,value in self.origin_tuples.items():
            tracks.append(value[2])
        self.origin_tracks = tracks

        min_len = 20
        max_len = 600
        min_dis = 100

        for track in tracks:

            if len(track) < max_len and len(track) > min_len:

                if math.sqrt((track[0][0]-track[-1][0])*(track[0][0]-track[-1][0]) + \
                    (track[0][1]-track[-1][1])*(track[0][1]-track[-1][1]))>min_dis:                
                    self.track_filtered.append(track)

        self.track_filtered = self.track_filtered[:self.thresh_num]

    def get_cluster(self, output_dir="."):
        tracks_clustered, _ = get_traj_cluster(self.track_filtered,min(\
            self.min_traj_num,len(self.track_filtered)),\
                int(0.5*len(self.track_filtered)))
        self.track_clustered = tracks_clustered
        print(len(tracks_clustered))
        if self.background is not None:
            colors = [(255,255,255),(0,0,0),(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,100,255),(0,255,255)]
            img_origin = self.background.copy()
            count = 0
            for track in self.origin_tracks:
                count += 1
                for point in track:
                    cv2.circle(img_origin,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,colors[count%8],2)
            cv2.imwrite(os.path.join(output_dir, "%s_trajs_origin.jpg" % self.video_name), img_origin)

            img_clustered = self.background.copy()
            count = 0
            for track in self.track_clustered:
                count += 1
                for point in track:
                    cv2.circle(img_clustered,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,colors[count%8],2)
        
            cv2.imwrite(os.path.join(output_dir, "%s_trajs_clustered_s.jpg" % self.video_name), img_clustered)

def track_cluster_from_labels(video_name, v_type="blazeit", cluster_min_num=16, cluster_tracks_num=96,
                               k=2000, save_result=True, label_dir=None, output_dir="./fixed_files/preprocessed",
                               mask_dir="./masks", d_type="train"):

    if label_dir is None:
        label_dir = "data/labels"

    if v_type == "blazeit":
        image_background = cv2.imread(os.path.join(mask_dir, "%s_mask.jpg" % video_name))
        label_parsed, tuple_dict, label_tuple_origin = get_blazeit_labels(video_name, d_type=d_type, label_dir=label_dir)
        if k > 0:
            sorted_keys = sorted(tuple_dict, key=tuple_dict.get)[:k]
            tuple_dict = {key: tuple_dict[key] for key in sorted_keys}

    else:
        image_background = cv2.imread("XXX.jpg"%(video_name))
        label_parsed, tuple_dict = get_m30_labels(video_name,k)

    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cluster = cluster_traj(video_name, image_background, tuple_dict, cluster_min_num, cluster_tracks_num)
    cluster.filter_traj()
    cluster.get_cluster(output_dir=save_dir)

    if save_result:
        np.save(os.path.join(save_dir, "%s_0_%d_tracks_clustered.npy" % (video_name, k)), np.array(cluster.track_clustered, dtype=object))
        if v_type == "blazeit":
            np.save(os.path.join(save_dir, "%s_0_%d_tracks_filtered.npy" % (video_name, k)), np.array(cluster.track_filtered, dtype=object))
        np.save(os.path.join(save_dir, "%s_0_%d_tracks_origin.npy" % (video_name, k)), np.array(cluster.origin_tracks, dtype=object))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Cluster trajectories from labels")
    parser.add_argument("--video_name", type=str, required=True, help="Video/dataset name (e.g. jnc0, caldot1)")
    parser.add_argument("--label_dir", type=str, default="/data/alenachao/data/leap-dataset/labels",
                        help="Base directory containing label files")
    parser.add_argument("--output_dir", type=str, default="./fixed_files/preprocessed",
                        help="Base directory to save clustered .npy files")
    parser.add_argument("--mask_dir", type=str, default="./masks",
                        help="Directory containing mask images")
    parser.add_argument("--d_type", type=str, default="train",
                        help="Data split to use for labels (train/test/val)")
    parser.add_argument("--cluster_min_num", type=int, default=16,
                        help="Minimum number of clusters")
    parser.add_argument("--cluster_tracks_num", type=int, default=96,
                        help="Max number of tracks to cluster")
    parser.add_argument("--k", type=int, default=0,
                        help="Max number of label tuples to use")
    args = parser.parse_args()

    track_cluster_from_labels(
        video_name=args.video_name,
        v_type="blazeit",
        cluster_min_num=args.cluster_min_num,
        cluster_tracks_num=args.cluster_tracks_num,
        k=args.k,
        save_result=True,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        d_type=args.d_type,
    )
