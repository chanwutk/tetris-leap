import yaml
import argparse
import torch
import numpy as np
import random
import os
import datetime
import json
import cv2
from loguru import logger
from tools.data_prepare import get_label_details
from settings.settings import video_details, Video_info
from settings import settings
from configs import cfg
from detector import Detector,Detector_origin
from retinanet_adapter import RetinaNetDetector
from inference import get_final_tuple_from_detector
import time
from evaluate import parse_intervals,evaluate_query_result,evaluated_object_recall

def getYaml(file_path):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def clear_dir(cfgs, args):

    import shutil
    out_dir = getattr(args, "output_dir", "./outputs")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(f'{out_dir}/selected_frames'):
        os.mkdir(f'{out_dir}/selected_frames')
    else:
        shutil.rmtree(f'{out_dir}/selected_frames')
        os.mkdir(f'{out_dir}/selected_frames')

    if not os.path.exists(f'{out_dir}/selected_frames_origin'):
        os.mkdir(f'{out_dir}/selected_frames_origin')
    else:
        shutil.rmtree(f'{out_dir}/selected_frames_origin')
        os.mkdir(f'{out_dir}/selected_frames_origin')

    if not os.path.exists(f'{out_dir}/traj_compute_tmp'):
        os.mkdir(f'{out_dir}/traj_compute_tmp')
    else:
        shutil.rmtree(f'{out_dir}/traj_compute_tmp')
        os.mkdir(f'{out_dir}/traj_compute_tmp')

    if not os.path.exists(f'{out_dir}/reid'):
        os.mkdir(f'{out_dir}/reid')
    if not os.path.exists(f'{out_dir}/reid/%s'%(cfgs["video_name"])):
        os.mkdir(f'{out_dir}/reid/%s'%(cfgs["video_name"]))
    else:
        shutil.rmtree(f'{out_dir}/reid/%s'%(cfgs["video_name"]))
        os.mkdir(f'{out_dir}/reid/%s'%(cfgs["video_name"]))

    if not os.path.exists(f'{out_dir}/traj_match'):
        os.mkdir(f'{out_dir}/traj_match')
    else:
        shutil.rmtree(f'{out_dir}/traj_match')
        os.mkdir(f'{out_dir}/traj_match')

def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/jackson.yaml')
    parser.add_argument('--weight', type=str, default='./weights/best.pt')
    parser.add_argument("-rw","--reid_weight",default=\
                            "XXX")

    parser.add_argument("--use_cluster",action="store_false") 
    parser.add_argument("--use_distill",action="store_false")
    parser.add_argument("--use_retinanet", action="store_true", help="Use RetinaNet detector")
    parser.add_argument("--use_filter",action="store_false")  
    parser.add_argument("--use_hand_label",action="store_false")
    parser.add_argument("--type",type=str,default="test")
    parser.add_argument("--adaptive_sample",action="store_false")
    parser.add_argument("--use_external_pattern",action="store_true")
    parser.add_argument("--cross_roads",action="store_true") 

    parser.add_argument("-vis","--visualize",action='store_false')  
    parser.add_argument("--active_log",action='store_true')
    parser.add_argument("-sr","--save_result",action="store_false",help="Save parsed result")
    parser.add_argument("-ld","--load",action='store_false',help="load file from dict")
    parser.add_argument("--use_label",action='store_true',help="Use preprocesed label instead of bytetracker")
    parser.add_argument("--use_mask",action='store_false',help="Use mask while parsing")
    parser.add_argument("--device",type=str,default='1')

    parser.add_argument("--video_dir", type=str, default=None,
                    help="Directory containing multiple videos to run on")
    parser.add_argument("--log_dir", type=str, default=".",
                    help="Directory to save the log files")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                    help="Directory to save the outputs")
    parser.add_argument("--motion_dir", type=str, default="./fixed_files/preprocessed",
                    help="Directory containing clustered motion pattern .npy files")

    return parser

def run_session(cfgs,args,logger):
    dataset_object = get_label_details(cfgs,args,logger)
    label_parsed, tuple_dict, label_tuple_origin = dataset_object.get_label()

    logger.info("%s %s %s Parsed"%(cfgs["full_name"],cfgs["video_name"],args.type))
    video_details.gt_labels = label_parsed
    video_details.gt_tuple = tuple_dict
    video_details.gt_tuple_origin = label_tuple_origin

    if not args.load and cfgs["dataset_group"]=='blazeit':  # 所有有真实标签的都不需要打伪标签
        
        from initialize import initialize
        tracks, preprocess_tuple = initialize(cfgs,args,logger)
        
    else:
        # try:
        #     if not args.use_external_pattern:
        #         tracks = np.load("./fixed_files/preprocessed/%s/"%(cfgs["video_name"])+cfgs["video_name"]+\
        #             "_0_%d_tracks_clustered.npy"%(cfgs["start_frame"]),allow_pickle=True)
        #     else:
        #         tracks = np.load("./fixed_files/preprocessed/%s/"%(cfgs["video_name"])+cfgs["video_name"]+\
        #             "_0_0_tracks_clustered.npy",allow_pickle=True)
                
        # except Exception:
        #     logger.info("False Path:"+"./fixed_files/preprocessed/"+\
        #         cfgs["video_name"]+"_0_%d_tracks_clustered.npy"%(cfgs["start_frame"]))
        #     exit()

        try:
            motion_base = os.path.join(args.motion_dir, cfgs["full_name"])
            if not args.use_external_pattern:
                tracks = np.load(os.path.join(motion_base, cfgs["full_name"]+\
                    "_0_%d_tracks_clustered.npy"%(cfgs["start_frame"])),allow_pickle=True)
            else:
                tracks = np.load(os.path.join(motion_base, cfgs["full_name"]+\
                    "_0_0_tracks_clustered.npy"),allow_pickle=True)
                
        except Exception:
            logger.info("False Path: "+os.path.join(args.motion_dir, cfgs["full_name"],
                cfgs["full_name"]+"_0_%d_tracks_clustered.npy"%(cfgs["start_frame"])))
            exit()

    if args.use_retinanet:
        logger.info("Using RetinaNet")
        detector = RetinaNetDetector(args.device, model_dir=os.path.dirname(args.weight))
    elif args.use_distill:
        logger.info("Distilled")
        detector = Detector(args.weight,640,args.device,cfgs['iou_thresh'],cfgs['conf_thresh'])
    else:
        logger.info("Origin")
        detector = Detector_origin(cfg)
        detector.preprocess(cfg, [cfgs["h"],cfgs["w"]])
    
    if args.video_dir is not None:
        video_path = cfgs["video_path"]
    elif cfgs["dataset_group"] == "blazeit":
        # video_path = os.path.join(settings.video_path,"standard_split",cfgs['full_name'],'concat',args.type,\
        #     '%s_%s.mp4'%(cfgs['video_name'],args.type))
        video_path = os.path.join(settings.video_path,cfgs["video_name"],'concat',args.type,\
            '%s_%s.mp4'%(cfgs['video_name'],args.type))
    elif cfgs["dataset_group"] == "m30":
        video_path = os.path.join("XXX","%s.mp4"%(cfgs["video_name"]))
    

    if cfgs["dataset_group"]=="blazeit":
        # args.reid_weight = os.path.join(args.reid_weight,cfgs['video_name'],"bagtricks_R50-ibn/model_best.pth")
        reid_weight = os.path.join(args.reid_weight,cfgs['full_name'],"model_final.pth")
    else:
        args.reid_weight = os.path.join(args.reid_weight,"jackson","bagtricks_R50-ibn/model_best.pth")

    frame_sampled, resolved_tuple = get_final_tuple_from_detector\
    (tracks,reid_weight,video_path,detector,cfgs,args,logger)
    # print("RESOVLED TUPLE: ", resolved_tuple[2:])

    
    
    logger.info("Sampled %d frames" % len(frame_sampled))
    logger.info("Filtered %d frames" % video_details.differencor)
    logger.info("Inference Time")
    logger.info(time.time()-video_details.start_time)
    logger.info("Detector Time")
    logger.info(video_details.detector_time)
    logger.info("Differencor Time")
    logger.info(video_details.frame_differencer_time)
    logger.info("Reid Time")
    logger.info(video_details.reid_time)
    logger.info("Match Time")
    logger.info(video_details.match_time)
    logger.info("Decode Time")
    logger.info(video_details.decode_time)

    logger.info("Video Path")
    logger.info(video_path)
    
    return_tuple = parse_intervals(resolved_tuple)

    if args.save_result:
        results = json.dumps(return_tuple)

        out_dir = getattr(args, "output_dir", "./outputs")

        os.makedirs(f'{out_dir}/parsed_results', exist_ok=True)
        os.makedirs(f'{out_dir}/parsed_results/{cfgs["full_name"]}', exist_ok=True)
        
        # f_outputs = open(f'{out_dir}/parsed_results/%s_results.json'%(cfgs['video_name']),'w')
        # f_outputs.write(results)
        # f_outputs.close()
        
        # f_outputs = open(f'{out_dir}/parsed_results/%s_results_dict.json'%(cfgs['video_name']),'w')
        # results = json.dumps(video_details.match_dict)
        # f_outputs.write(results)
        # f_outputs.close()

        tracks_list = [t.tolist() if hasattr(t, "tolist") else t for t in tracks]

        # with open(f'{out_dir}/parsed_results/%s_tracks.json'%(cfgs['video_name']), "w") as f:
        #     json.dump(tracks_list, f, indent=2)


        # output_prefix = f"{cfgs['full_name']}_{cfgs['video_name']}"

        with open(f'{out_dir}/parsed_results/{cfgs["full_name"]}/{cfgs["video_name"]}_results.json', 'w') as f:
            f.write(results)

        with open(f'{out_dir}/parsed_results/{cfgs["full_name"]}/{cfgs["video_name"]}_results_dict.json', 'w') as f:
            json.dump(video_details.match_dict, f, indent=2)

        with open(f'{out_dir}/parsed_results/{cfgs["full_name"]}/{cfgs["video_name"]}_tracks.json', "w") as f:
            json.dump(tracks_list, f, indent=2)

        speed_path = f'{out_dir}/parsed_results/{cfgs["full_name"]}/speed.tsv'
        os.makedirs(os.path.dirname(speed_path), exist_ok=True)

        # Collect metrics
        total_time = time.time() - video_details.start_time
        det_time = video_details.detector_time
        diff_time = video_details.frame_differencer_time
        reid_time = video_details.reid_time
        match_time = video_details.match_time
        decode_time = video_details.decode_time
        sampled = len(frame_sampled)
        filtered = video_details.differencor
        video = cfgs["video_name"]

        # Determine if we need a header
        write_header = not os.path.exists(speed_path)

        with open(speed_path, "a") as f:
            if write_header:
                f.write(
                    "video_name\tinference_total\t"
                    "detector\tdifferencer\treid\tmatch\tdecode\t"
                    "frames_sampled\tframes_filtered\n"
                )
            
            f.write(
                f"{video}\t{total_time:.4f}\t"
                f"{det_time:.4f}\t{diff_time:.4f}\t{reid_time:.4f}\t"
                f"{match_time:.4f}\t{decode_time:.4f}\t"
                f"{sampled}\t{filtered}\n"
            )
        
    video_details.return_tuple = return_tuple

    evaluate_query_result(args,cfgs,label_tuple_origin)
    recall_by_sampled_frame = evaluated_object_recall(video_details.frame_sampled,tuple_dict)

    logger.info("Vehicle Recall: ")
    logger.info(recall_by_sampled_frame)

#     attrs_to_dump = {k: getattr(video_details, k) for k in [
#     "traj_dict", "allocate_id", "traj_type_dict", "gt_labels", "gt_tuple",
#     "gt_tuple_origin", "match_dict", "object_type", "return_tuple",
#     "stop_cars", "frame_sampled", "bbox_rate",
#     "stop_area", "skip_frames", "adaptive_skip", "direction", "reverse"
# ]}

#     with open("video_details_summary.json", "w") as f:
#         json.dump(attrs_to_dump, f, indent=2)

    cleaned_resolved_tuple = {}

    for key, value in video_details.resolved_tuple.items():
        # value is a list of [detections, relations, patch]
        if isinstance(value, list) and len(value) >= 2:
            cleaned_resolved_tuple[key] = [value[0], value[1], value[3]]  # keep index 0,1, and 3
        else:
            cleaned_resolved_tuple[key] = value
        
    # with open("resolved_tuple_clean.json", "w") as f:
    #     json.dump(cleaned_resolved_tuple, f, indent=2)
    
# if __name__ =="__main__":

#     parser = make_parser()
#     args = parser.parse_args()
#     cfgs = getYaml(args.config)
#     logger.add('%s.log'%(datetime.datetime.now().strftime('%Y-%m-%d_%H')))
#     set_seed(cfgs['seed'])
#     clear_dir(cfgs)
#     logger.info("Clean Folders")
#     run_session(cfgs,args,logger)

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    cfgs = getYaml(args.config)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger.add(os.path.join(args.log_dir, '%s.log' % (datetime.datetime.now().strftime('%Y-%m-%d_%H'))))
    
    set_seed(cfgs['seed'])
    clear_dir(cfgs, args)
    logger.info("Clean Folders")

    # --- MULTI-VIDEO MODE ---
    if args.video_dir is not None:
        video_files = sorted([
            f for f in os.listdir(args.video_dir)
            if f.endswith(".mp4")
        ])

        logger.info(f"Found {len(video_files)} videos in {args.video_dir}")

        for f in video_files:
            video_name = os.path.splitext(f)[0]  # e.g. "0", "1", "2"
            logger.info(f"Processing video: {video_name}")

            # override cfgs["video_name"] for this run
            cfgs["video_name"] = video_name  

            # path to the actual video file
            video_path = os.path.join(args.video_dir, f)

            # store this temporary path so run_session can use it
            cfgs["video_path"] = video_path

            run_session(cfgs, args, logger)

        exit()
    
    # --- SINGLE VIDEO (original behavior) ---
    run_session(cfgs, args, logger)