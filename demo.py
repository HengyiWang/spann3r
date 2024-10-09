import os
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from dust3r.losses import L21
from spann3r.model import Spann3R
from dust3r.inference import inference
from dust3r.utils.geometry import geotrf
from dust3r.image_pairs import make_pairs
from spann3r.loss import Regr3D_t_ScaleShiftInv
from spann3r.datasets import *
from torch.utils.data import DataLoader
from spann3r.tools.eval_recon import accuracy, completion
from spann3r.tools.vis import render_frames, find_render_cam, vis_pred_and_imgs
from pose_utils import solve_cemara

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

    return parser

@torch.no_grad()
def main(args):

    workspace = args.save_path
    os.makedirs(workspace, exist_ok=True)

    ##### Load model
    model = Spann3R(dus3r_name='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', 
                use_feat=False).to(args.device)
    
    model.load_state_dict(torch.load(args.ckpt_path)['model'])
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
    conf_sig_all = []
    cameras_all = []

    for j, view in enumerate(ordered_batch):
        
        image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
        mask = view['valid_mask'].cpu().numpy()[0]

        pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
        conf = preds[j]['conf'][0].cpu().data.numpy()
        conf_sig = (conf - 1) / conf
        pts_gt = view['pts3d'].cpu().numpy()[0]

        camera = solve_cemara(torch.tensor(pts), torch.tensor(conf_sig) > args.conf_thresh, args.device)
        
        images_all.append((image[None, ...] + 1.0)/2.0)
        pts_all.append(pts[None, ...])
        pts_gt_all.append(pts_gt[None, ...])
        masks_all.append(mask[None, ...])
        conf_sig_all.append(conf_sig[None, ...])
        cameras_all.append(camera)

    images_all = np.concatenate(images_all, axis=0)
    pts_all = np.concatenate(pts_all, axis=0)
    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)
    conf_sig_all = np.concatenate(conf_sig_all, axis=0)

    save_params = dict(
        images_all=images_all,
        pts_all=pts_all,
        pts_gt_all=pts_gt_all,
        masks_all=masks_all,
        conf_sig_all=conf_sig_all
        )
    
    np.save(os.path.join(save_demo_path, f"{demo_name}.npy"), save_params)


    # Save point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all[conf_sig_all>args.conf_thresh].reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(images_all[conf_sig_all>args.conf_thresh].reshape(-1, 3))
    o3d.io.write_point_cloud(os.path.join(save_demo_path, f"{demo_name}_conf{args.conf_thresh}.ply"), pcd)


    if args.vis:
        render_frames(pts_all, images_all, cameras_all, save_demo_path, mask=conf_sig_all>args.conf_thresh)
        vis_pred_and_imgs(pts_all, save_demo_path, images_all=images_all, conf_all=conf_sig_all)



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)