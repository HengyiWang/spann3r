import os
import sys
import math
import json
import time
import torch
import argparse
import datetime
import numpy as np
import torch.backends.cudnn as cudnn

import croco.utils.misc as misc 

from pathlib import Path
from typing import Sized
from shutil import copyfile
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from spann3r.model import Spann3R
from dust3r.losses import L21
from spann3r.datasets import *
from spann3r.loss import Regr3D_t, ConfLoss_t, Regr3D_t_ScaleShiftInv
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('Spann3R training', add_help=False)
    parser.add_argument('--model', default="Spann3R(dus3r_name='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', use_feat=False, mem_pos_enc=False)",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    
    # Loss
    parser.add_argument('--train_criterion', 
                        default="ConfLoss_t(Regr3D_t(L21, norm_mode='avg_dis', fix_first=False), alpha=0.4)",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default="Regr3D_t_ScaleShiftInv(L21, gt_scale=True)")

    # Datasets
    parser.add_argument('--train_dataset', 
                        default= "10000 @ Co3d(split='train', ROOT='./data/co3d_preprocessed_50', resolution=224, num_frames=5, mask_bg='rand', transform=ColorJitter) + 10000 @ Co3d(split='train', ROOT='./data/co3d_preprocessed_50', resolution=224, num_frames=5, mask_bg='rand', transform=ColorJitter, use_comb=False) + 10000 @ BlendMVS(split='train', ROOT='./data/blendmvg', resolution=224) + 10000 @ Scannetpp(split='train', ROOT='./data/scannetpp', resolution=224, transform=ColorJitter) + 10000 @ habitat(split='train', ROOT='./data/habitat_5frame', resolution=224, transform=ColorJitter) + 10000 @ Scannet(split='train', ROOT='./data/scannet', resolution=224, transform=ColorJitter, max_thresh=50) + 10000 @ ArkitScene(split='train', ROOT='./data/arkit_lowres', resolution=224, transform=ColorJitter, max_thresh=100)",
                        required=False, type=str, help="training set")
    parser.add_argument('--test_dataset', 
                        default="Scannetpp(split='val', ROOT='./data/scannetpp', resolution=224, num_seq=1, kf_every=10, seed=777, full_video=True) + 1000 @ Co3d(split='test', ROOT='./data/co3d_preprocessed_50', resolution=224, num_frames=5, mask_bg=False, seed=777)", 
                        type=str, help="testing set")
    
     # Exp
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    
    # Training
    parser.add_argument('--batch_size', default=2, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--batch_size_test', default=1, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=120, type=int, help="Maximum number of epochs for the scheduler")
    
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-06, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    
    # others
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_workers_test', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=5, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')
    
    parser.add_argument('--alpha_c2f', type=int, default=1, help='use alpha c2f')
    
    # output dir 
    parser.add_argument('--output_dir', default='./output/all_alpha04_lr05', type=str, help="path where to save the output")
    
    return parser
    
@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    
    save_path = os.path.join(args.output_dir, f'eval_{epoch}')
        
    os.makedirs(save_path, exist_ok=True)

    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        for view in batch:
            for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
        
        
        preds, preds_all = model.forward(batch)
        
        if i < 100:
            images_all = []
            pts_all = []
            for j, view in enumerate(batch):
                img_idx = 0
                mask = view['depthmap'][img_idx:img_idx+1].cpu().numpy()!=0
                image = view['img'][img_idx:img_idx+1].permute(0, 2, 3, 1).cpu().numpy()[mask].reshape(-1, 3)
                
                pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'][img_idx:img_idx+1].detach().cpu().numpy()
                pts = pts[mask].reshape(-1, 3)
                
                images_all.append(image)
                pts_all.append(pts)
            images_all = np.concatenate(images_all, axis=0)
            


            pts_all = np.concatenate(pts_all, axis=0)
            # create open3d point cloud and save
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_all.reshape(-1, 3))
            pcd.colors = o3d.utility.Vector3dVector((images_all.reshape(-1, 3)+1.0)/2.0)
            o3d.io.write_point_cloud(os.path.join(save_path, view['dataset'][0]+f"_idx_{i}.ply"), pcd)
            
        
        loss, loss_details, loss_factor = criterion.compute_frame_loss(batch, preds_all)
        loss_value = float(loss)
        
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix+'_'+name, val, 1000*epoch)

    return results

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
        
    epoch_ratio = epoch/args.epochs
    if epoch_ratio < 0.75:
        active_ratio = min(1, epoch/args.epochs*2.0)
    else:
        active_ratio = max(0.5, 1 - (epoch_ratio - 0.75) / 0.25)
    data_loader.dataset.set_ratio(active_ratio)
    #print(f"active thresh: {data_loader.datasets.dataset.active_thresh}")
    
    
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)
        
        for view in batch:
            for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
        
        
        preds, preds_all = model.forward(batch)
        loss, loss_details, loss_factor = criterion.compute_frame_loss(batch, preds_all)
        loss += loss_factor     

        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=1.0) # 
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            log_writer.add_scalar('active_ratio', active_ratio, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_'+name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    print("output_dir: "+args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)

    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size_test, args.num_workers_test, test=True)
                        for dataset in args.test_dataset.split('+')}
    
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    
    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    test_criterion = eval(args.test_criterion).to(device)

    alpha_init = train_criterion.alpha
    
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory
    
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module
    
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update({test_name+'_'+k: v for k, v in test_stats[test_name].items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)
    
    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None
    
    file_path_all =[ './']
        
    os.makedirs(os.path.join(args.output_dir, 'recording'), exist_ok=True)
    
    for file_path in file_path_all:
        cur_dir = os.path.join(args.output_dir, 'recording', file_path)
        os.makedirs(cur_dir, exist_ok=True)
        
        files = os.listdir(file_path)
        for f_name in files:
            if f_name[-3:] == '.py':
                copyfile(os.path.join(file_path, f_name), os.path.join(cur_dir, f_name))

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs+1):
        
        # TODO: Save last check point
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch-1, 'last', best_so_far)
        
        # Test on multiple datasets
        new_best = False
        if (epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0):
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(model, test_criterion, testset,
                                       device, epoch, log_writer=log_writer, args=args, prefix=test_name)
                test_stats[test_name] = stats

                # Save best of all
                if stats['loss_med'] < best_so_far:
                    best_so_far = stats['loss_med']
                    new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)
        
        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch-1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch-1, 'best', best_so_far)
            
        if epoch >= args.epochs:
            break 

        if args.alpha_c2f:
            train_criterion.alpha = alpha_init - 0.2 * max((epoch - 0.5 * args.epochs) / (0.5 * args.epochs), 0)
            print('Update alpha to', train_criterion.alpha)
        
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    