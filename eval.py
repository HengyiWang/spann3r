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

def get_args_parser():
    parser = argparse.ArgumentParser('Spann3R evaluation', add_help=False)
    parser.add_argument('--exp_path', type=str, help='Path to experiment folder', default='./checkpoints')
    parser.add_argument('--exp_name', type=str, default='ckpt_best', help='Path to experiment folder')
    parser.add_argument('--ckpt', type=str, default='spann3r.pth', help='ckpt name')
    parser.add_argument('--scenegraph_type', type=str, default='complete', help='scenegraph type')
    parser.add_argument('--offline', action='store_true', help='offline reconstruction')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--conf_thresh', type=float, default=0.0, help='confidence threshold')

    return parser
    

def main(args):
    workspace = args.exp_path
    ckpt_path = osp.join(workspace, args.ckpt)
    if not osp.exists(workspace):
        raise FileNotFoundError(f"Workspace {workspace} not found")
    
    exp_path = osp.join(workspace, args.exp_name)
    os.makedirs(exp_path, exist_ok=True)

    datasets_all = {
        '7scenes': SevenScenes(split='test', ROOT="./data/7scenes",
                                resolution=224, num_seq=1, full_video=True, kf_every=20),
        'NRGBD': NRGBD(split='test', ROOT="./data/neural_rgbd", 
                           resolution=224, num_seq=1, full_video=True, kf_every=40),
        'DTU': DTU(split='test', ROOT="./data/dtu_test",
                   resolution=224, num_seq=1, full_video=True, kf_every=5),
    }
    model = Spann3R(dus3r_name='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', 
                use_feat=False).to(args.device)
    
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device)['model'])
    model.eval()    


    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(exp_path, name_data)
            if args.offline:
                save_path = osp.join(save_path + '_offline')
            os.makedirs(save_path, exist_ok=True)

            log_file = osp.join(save_path, 'logs.txt')
            os.makedirs(save_path, exist_ok=True)

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            for i, batch in enumerate(dataloader):

                for view in batch:
                    for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                        if name not in view:
                            continue
                        view[name] = view[name].to(args.device, non_blocking=True)


                print(f'Started reconstruction for {name_data} {i+1}/{len(dataloader)}')
                
                if args.offline:
                    imgs_all = []
                    for j, view in enumerate(batch):
                        img = view['img']
                        shape1 = [img.size()[::-1]]

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

                print(f'Finished reconstruction for {name_data} {i+1}/{len(dataloader)}, FPS: {fps:.2f}')

                fps_all.append(fps)
                time_all.append(end - start)
                

                # Evaluation
                print(f'Evaluation for {name_data} {i+1}/{len(dataloader)}')
                gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = criterion.get_all_pts3d_t(ordered_batch, preds_all)
                pred_scale, gt_scale, pred_shift_z, gt_shift_z  = monitoring['pred_scale'], monitoring['gt_scale'], monitoring['pred_shift_z'], monitoring['gt_shift_z']
                
                in_camera1 = None
                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []
                conf_all = []

                for j, view in enumerate(ordered_batch):
                    if in_camera1 is None:
                        in_camera1 = view['camera_pose'][0].cpu()
                    
                    image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
                    mask = view['valid_mask'].cpu().numpy()[0]

                    # pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
                    pts = pred_pts[0][j].cpu().numpy()[0] if j < len(pred_pts[0]) else pred_pts[1][-1].cpu().numpy()[0]
                    conf = preds[j]['conf'][0].cpu().data.numpy()

                    pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                    #### Align predicted 3D points to the ground truth 
                    pts[..., -1] += gt_shift_z.cpu().numpy().item()
                    pts = geotrf(in_camera1, pts)

                    pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                    pts_gt = geotrf(in_camera1, pts_gt)

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

                scene_id = view['label'][0].rsplit('/', 1)[0]

                save_params = {}

                save_params['images_all'] = images_all
                save_params['pts_all'] = pts_all
                save_params['pts_gt_all'] = pts_gt_all
                save_params['masks_all'] = masks_all
                save_params['conf_all'] = conf_all

                np.save(os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"), save_params)


                if 'DTU' in name_data:
                    threshold = 100
                else:
                    threshold = 0.1

                
                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_all > 0]
                images_all_masked = images_all[masks_all > 0]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_all_masked.reshape(-1, 3))
                pcd.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3))
                o3d.io.write_point_cloud(os.path.join(save_path, f"{scene_id.replace('/', '_')}-mask.ply"), pcd)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked.reshape(-1, 3))
                pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3) / 255.0)
                o3d.io.write_point_cloud(os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"), pcd_gt)

                trans_init = np.eye(4)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd, pcd_gt, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                
                transformation = reg_p2p.transformation
                            
                pcd = pcd.transform(transformation)
                pcd.estimate_normals()
                pcd_gt.estimate_normals()

                gt_normal = np.asarray(pcd_gt.normals)
                pred_normal = np.asarray(pcd.normals)

                acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
                comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
                

                print(f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}", file=open(log_file, "a"))

                acc_all += acc
                comp_all += comp
                nc1_all += nc1
                nc2_all += nc2

                acc_all_med += acc_med
                comp_all_med += comp_med
                nc1_all_med += nc1_med
                nc2_all_med += nc2_med


                # release cuda memory
                torch.cuda.empty_cache()

                print(f"Finished evaluation for {name_data} {i+1}/{len(dataloader)}")



                # Get depth from pcd and run TSDFusion
                
            
            print(f"Dataset: {name_data}, Accuracy: {acc_all/len(dataloader)}, Completion: {comp_all/len(dataloader)}, NC1: {nc1_all/len(dataloader)}, NC2: {nc2_all/len(dataloader)} - Acc_med: {acc_all_med/len(dataloader)}, Comp_med: {comp_all_med/len(dataloader)}, NC1_med: {nc1_all_med/len(dataloader)}, NC2_med: {nc2_all_med/len(dataloader)}", file=open(log_file, "a"))
            print(f"Average fps: {sum(fps) / len(fps)}, Average time: {sum(time_all) / len(time_all)}", file=open(log_file, "a"))
                


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
                








                    
                    



                    





                

