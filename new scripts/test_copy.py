from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--boxes_sammed", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=False, help="save reslut")
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
    
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def main(args):
    print('*' * 100)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('*' * 100)

    model = sam_model_registry[args.model_type](args).to(args.device)

    criterion = FocalDiceloss_IoULoss()
    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    test_metrics_lists = [[] for _ in args.metrics]  # Create a list of lists to store scores for each metric
    prompt_dict = {}
    # ----------------------------------------
    # Define the CSV file path
    csv_file_path = os.path.join(args.work_dir, args.run_name, "metrics.csv")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['Mask Path', 'IOU', 'Dice Score'])

        for i, batched_input in enumerate(test_pbar):
            
            batched_input = to_device(batched_input, args.device)
            ori_labels = batched_input["ori_label"]
            original_size = batched_input["original_size"]
            labels = batched_input["label"]
            img_name = batched_input['name'][0]
            if args.prompt_path is None:
                prompt_dict[img_name] = {
                            "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                            "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                            "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                            }

            with torch.no_grad():
                image_embeddings = model.image_encoder(batched_input["image"])

            if args.boxes_prompt:
                save_path = os.path.join(args.work_dir, args.run_name, "boxes_sammed")
                batched_input["point_coords"], batched_input["point_labels"] = None, None
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                points_show = None

            else:
                save_path = os.path.join(f"{args.work_dir}", args.run_name, f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
                batched_input["boxes"] = None
                point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
        
                for iter in range(args.iter_point):
                    masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                    if iter != args.iter_point-1:
                        batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                        batched_input = to_device(batched_input, args.device)
                        point_coords.append(batched_input["point_coords"])
                        point_labels.append(batched_input["point_labels"])
                        batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                        batched_input["point_labels"] = torch.concat(point_labels, dim=1)
    
                points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

            masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
            if args.save_pred:
                save_masks(masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show, visual_prompt=True)
                # setting visual_prompt = True

            loss = criterion(masks, ori_labels, iou_predictions)
            test_loss.append(loss.item())

            test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
            test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

            # Print individual metrics for each image
            print(f"Metrics for {img_name}: {dict(zip(args.metrics, test_batch_metrics))}")

            # --------------------------------------------------------------------
            # Assuming you calculate or retrieve img_name, iou, dice_score within this loop
            mask_path = os.path.join(save_path, img_name)  # Modify as needed
            iou = test_batch_metrics[0]  # Assuming IOU is the first metric
            dice_score = test_batch_metrics[1]  # Assuming Dice Score is the second metric

            # Save metrics to the CSV file for each image
            writer.writerow([img_name, iou, dice_score])

            # -------------------------------------------------------------------
            for j, metric in enumerate(test_batch_metrics):
                test_metrics_lists[j].append(metric)  # Append individual metric scores to their respective lists

        average_loss = np.mean(test_loss)
        # Calculate mean and standard deviation for each metric
        test_metrics_mean = {args.metrics[i]: np.mean(test_metrics_lists[i]) for i in range(len(args.metrics))}
        test_metrics_std = {args.metrics[i] + "_std": np.std(test_metrics_lists[i], ddof=1) for i in range(len(args.metrics))}  # ddof=1 for sample standard deviation

        # if args.prompt_path is None:
        #     with open(os.path.join(args.work_dir, args.run_name, "prompt.json"), 'w') as f:
        #         json.dump(prompt_dict, f, indent=4)

        # og code:
        if args.prompt_path is None:
            with open(os.path.join(args.work_dir,f'{args.image_size}_prompt.json'), 'w') as f:
                json.dump(prompt_dict, f, indent=2)

        # Print aggregated metrics with standard deviation
        print(f"Test loss: {average_loss:.4f}")
        print("Aggregated metrics (Mean ± Std):")
        for metric in args.metrics:
            print(f"{metric}: {test_metrics_mean[metric]:.4f} ± {test_metrics_std[metric + '_std']:.4f}")


if __name__ == '__main__':
    args = parse_args()
    args.device = 'cpu'
    # args.work_dir = r"C:\\Users\\nguyen\\Documents\\BMIF_project\\SAM_Med2D"
    args.work_dir = "/home/vivian/SAM_Med2D"
    args.save_pred = False
    args.boxes_prompt = False
    args.point_num = 1
    args.iter_point = 1
    # -----------------------------------------------
    # data split by patient
    args.data_path = "patient_split/rf_data/test"
    # args.data_path = "patient_split/bm_data/test"
    # args.data_path = 'RF_v2/test'

    # args.data_path = 'patient_split/cross_val_rf/val2/test'
    # args.prompt_path = 'patient_split/bm_data/test_pt_bm.json'

    # args.prompt_path = 'patient_split/rf_data/test_pt_rf.json'
    args.prompt_path = 'all_pt_rf.json'
    # args.sam_checkpoint = "models/patient_BM_encoder/epoch5_sam.pth" # patient rf ft encoder  
    # args.sam_checkpoint = "models/patient_RF_encoder/epoch3_sam.pth"  
    # args.sam_checkpoint = '/DATA/vivian/models/patient_RF_no_adapter/epoch9_sam.pth' # RF no adapter
    # args.sam_checkpoint = '/DATA/vivian/models/patient_BM_no_adapter/epoch7_sam.pth' # BM no adapter 
    # args.sam_checkpoint = '/DATA/vivian/models/patient_RFBM_encoder/epoch5_sam.pth' # RFBM adapter
    # args.sam_checkpoint = '/DATA/vivian/models/patient_RFBM_no_adapter/epoch7_sam.pth' # RFBM no adapter
    args.sam_checkpoint = '/DATA/vivian/models/patient_RFv2_adapter/epoch6_sam.pth' # RFv2
    # args.sam_checkpoint = '/DATA/vivian/models/patient_RFv2_no_adapter/epoch9_sam.pth' # RFv2 no adapter
    # args.sam_checkpoint = '/DATA/vivian/models/patient_RFv3_adapter/epoch2_sam.pth' # RFv3
    # args.sam_checkpoint = '/DATA/vivian/models/patient_RFv3_no_adapter/epoch6_sam.pth' # RFv3 no adapter
    # args.sam_checkpoint = '/DATA/vivian/models/RF_crossval2_adapter/epoch6_sam.pth' #cross val
    args.run_name = "final_masks/SAM_rfv2_no_adapter"
    args.encoder_adapter = False 

    main(args)

