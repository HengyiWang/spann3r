import os
import cv2
import json
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp

from torch.utils.data import DataLoader

from dust3r.losses import L21
from dust3r.utils.geometry import inv
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import imread_cv2
from dust3r.post_process import estimate_focal_knowing_depth

from spann3r.datasets import *
from spann3r.model import Spann3R
from spann3r.loss import Regr3D_t_ScaleShiftInv
from spann3r.tools.eval_recon import accuracy, completion
from spann3r.tools.vis import render_frames, find_render_cam, vis_pred_and_imgs

def get_args_parser():
    parser = argparse.ArgumentParser('Spann3R demo', add_help=False)
    parser.add_argument('--save_path', type=str, default='./output/demo/', help='Path to experiment folder')
    parser.add_argument('--demo_path', type=str, default='./examples/s00567', help='Path to experiment folder')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/spann3r.pth', help='ckpt path')
    parser.add_argument('--scenegraph_type', type=str, default='complete', help='scenegraph type')
    parser.add_argument('--offline', action='store_true', help='offline reconstruction')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--conf_thresh', type=float, default=1e-3, help='confidence threshold')
    parser.add_argument('--kf_every', type=int, default=10, help='map every kf_every frames')
    parser.add_argument('--vis', action='store_true', help='visualize')
    parser.add_argument('--vis_cam', action='store_true', help='visualize camera pose')
    parser.add_argument('--save_ori', action='store_true', help='save original parameters for NeRF')
    parser.add_argument('--dynamic', action='store_true', help='dynamic mode')

    return parser

def get_transform_json(H, W, focal, poses_all, ply_file_path, ori_path=None):
    transform_dict = {
        'w': W,
        'h': H,
        'fl_x': focal.item(),
        'fl_y': focal.item(),
        'cx': W/2,
        'cy': H/2,
        'k1': 0,
        'k2': 0,
        'p1': 0,
        'p2': 0,
        'camera_model': 'OPENCV',
    }
    frames = []

    for i, pose in enumerate(poses_all):
        # CV2 GL format
        pose[:3, 1] *= -1
        pose[:3, 2] *= -1
        frame = {
            'file_path': f"imgs/img_{i:04d}.png" if ori_path is None else ori_path[i],
            'transform_matrix': pose.tolist()
        }
        frames.append(frame)
    
    transform_dict['frames'] = frames
    transform_dict['ply_file_path'] = ply_file_path

    return transform_dict

@torch.no_grad()
def main(args):

    workspace = args.save_path
    os.makedirs(workspace, exist_ok=True)

    ##### Load model
    model = Spann3R(dus3r_name='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', 
                use_feat=False).to(args.device)
    
    model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device)['model'])
    model.eval()

    ##### Load dataset
    dataset = Demo(ROOT=args.demo_path, resolution=224, full_video=True, kf_every=args.kf_every)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = dataloader.__iter__().__next__()

    ##### Inference
    for view in batch:
         view['img'] = view['img'].to(args.device, non_blocking=True)
           

    demo_name = args.demo_path.split("/")[-1]

    print(f'Started reconstruction for {demo_name}')

    if args.offline:
        imgs_all = []
        for j, view in enumerate(batch):
            img = view['img']
            imgs_all.append(
                dict(
                    img=img,
                    true_shape=torch.tensor(img.shape[2:]).unsqueeze(0),
                    idx=j,
                    instance=str(j)
                )
            )
        start = time.time()

        pairs = make_pairs(imgs_all, scene_graph=args.scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, model.dust3r, args.device, batch_size=2, verbose=True)
        preds, preds_all, idx_used = model.offline_reconstruction(batch, output) 
        
        end = time.time()

        ordered_batch = [batch[i] for i in idx_used]
    else:
        start = time.time()
        preds, preds_all = model.forward(batch) 
        end = time.time()
        ordered_batch = batch
        
    fps = len(batch) / (end - start)
    

    print(f'Finished reconstruction for {demo_name}, FPS: {fps:.2f}')

    ##### Save results

    save_demo_path = osp.join(workspace, demo_name)
    os.makedirs(save_demo_path, exist_ok=True)

    pts_all = []
    pts_gt_all = []
    images_all = []
    masks_all = []
    conf_all = []
    poses_all = []


    ##### estimate focal length
    _, H, W, _ = preds[0]['pts3d'].shape
    pp = torch.tensor((W/2, H/2))
    focal = estimate_focal_knowing_depth(preds[0]['pts3d'].cpu(), pp, focal_mode='weiszfeld')
    print(f'Estimated focal of first camera: {focal.item()} (224x224)')

    intrinsic = np.eye(3)
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[:2, 2] = pp


    for j, view in enumerate(ordered_batch):
        
        image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
        mask = view['valid_mask'].cpu().numpy()[0]

        pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
        conf = preds[j]['conf'][0].cpu().data.numpy()

        pts_gt = view['pts3d'].cpu().numpy()[0]

        ##### Solve PnP-RANSAC
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        points_2d = np.stack((u, v), axis=-1)
        dist_coeffs = np.zeros(4).astype(np.float32)
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            pts.reshape(-1, 3).astype(np.float32), 
            points_2d.reshape(-1, 2).astype(np.float32), 
            intrinsic.astype(np.float32), 
            dist_coeffs)
    
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extrinsic parameters (4x4 matrix)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

        poses_all.append(inv(extrinsic_matrix))
        images_all.append((image[None, ...] + 1.0)/2.0)
        pts_all.append(pts[None, ...])
        pts_gt_all.append(pts_gt[None, ...])
        masks_all.append(mask[None, ...])
        conf_all.append(conf[None, ...])
    
    images_all = np.concatenate(images_all, axis=0)
    pts_all = np.concatenate(pts_all, axis=0)
    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)
    conf_all = np.concatenate(conf_all, axis=0)
    poses_all = np.stack(poses_all, axis=0)

    save_params = dict(
        images_all=images_all,
        pts_all=pts_all,
        pts_gt_all=pts_gt_all,
        masks_all=masks_all,
        conf_all=conf_all,
        poses_all=poses_all,
        intrinsic=intrinsic,
        )
    
    np.save(os.path.join(save_demo_path, f"{demo_name}.npy"), save_params)


    # Save point cloud
    conf_sig_all = (conf_all-1) / conf_all

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all[conf_sig_all>args.conf_thresh].reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(images_all[conf_sig_all>args.conf_thresh].reshape(-1, 3))
    o3d.io.write_point_cloud(os.path.join(save_demo_path, f"{demo_name}_conf{args.conf_thresh}.ply"), pcd)


    if args.vis:
        camera_parameters = find_render_cam(pcd, poses_all if args.vis_cam else None)

        render_frames(pts_all, images_all, camera_parameters, save_demo_path, mask=conf_sig_all>args.conf_thresh, dynamic=args.dynamic)
        vis_pred_and_imgs(pts_all, save_demo_path, images_all=images_all, conf_all=conf_sig_all)
    
    # Save transform.json
    if args.save_ori:
        scale_factor = ordered_batch[0]['camera_intrinsics'][0, 0, 0]
        assert scale_factor < 1.0, "Scale factor should be less than 1.0"
        focal_ori = focal / scale_factor

        image = imread_cv2(ordered_batch[0]['label'][0])

        H_ori, W_ori = image.shape[:2]

        paths_all = [osp.normpath(osp.join(osp.abspath(os.getcwd()), view['label'][0]))
                      for view in ordered_batch]

        transform_dict = get_transform_json(H_ori, W_ori, focal_ori, poses_all, 
                                            f"{demo_name}_conf{args.conf_thresh}.ply",
                                            ori_path=paths_all)


        
    
    else:
        transform_dict = get_transform_json(H, W, focal, poses_all, f"{demo_name}_conf{args.conf_thresh}.ply")
    

    # Save to json
    with open(os.path.join(save_demo_path, 'transforms.json'), 'w') as f:
        json.dump(transform_dict, f, indent=4)




if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)