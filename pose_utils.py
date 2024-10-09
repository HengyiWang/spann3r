import numpy as np
import torch
import cv2
import open3d as o3d
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import inv

def estimate_focal(pts3d_i, pp=None):
    if pp is None:
        H, W, THREE = pts3d_i.shape
        assert THREE == 3
        pp = torch.tensor((W/2, H/2), device=pts3d_i.device)
    focal = estimate_focal_knowing_depth(pts3d_i.unsqueeze(0), pp.unsqueeze(0), focal_mode='weiszfeld').ravel()
    return float(focal)

def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)

def sRT_to_4x4(scale, R, T, device):
    trf = torch.eye(4, device=device)
    trf[:3, :3] = R * scale
    trf[:3, 3] = T.ravel()  # doesn't need scaling
    return trf

def to_numpy(tensor):
    return tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10):
    # extract camera poses and focals with RANSAC-PnP
    if msk.sum() < 4:
        return None  # we need at least 4 points for PnP
    pts3d, msk = map(to_numpy, (pts3d, msk))

    H, W, THREE = pts3d.shape
    assert THREE == 3
    pixels = pixel_grid(H, W)

    if focal is None:
        S = max(W, H)
        tentative_focals = np.geomspace(S/2, S*3, 21)
    else:
        tentative_focals = [focal]

    if pp is None:
        pp = (W/2, H/2)
    else:
        pp = to_numpy(pp)

    best = 0,
    for focal in tentative_focals:
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

        success, R, T, inliers = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                                    iterationsCount=niter_PnP, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)

        if not success:
            continue

        score = len(inliers)
        if success and score > best[0]:
            best = score, R, T, focal

    if not best[0]:
        return None

    _, R, T, best_focal = best
    R = cv2.Rodrigues(R)[0]  # world to cam
    R, T = map(torch.from_numpy, (R, T))

    return best_focal, inv(sRT_to_4x4(1, R, T, device))  # cam to world

def solve_cemara(pts3d, msk, device, pp=None):
    # Estimate focal length
    focal = estimate_focal(pts3d, pp)
    
    # Compute camera pose using PnP
    result = fast_pnp(pts3d, focal, msk, device, pp)
    
    if result is None:
        return None, None, None
    
    best_focal, camera_to_world = result
    
    # Construct K matrix
    H, W, _ = pts3d.shape
    if pp is None:
        pp = (W/2, H/2)
    
    camera_parameters = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(W, H, 
                             best_focal, best_focal, 
                             pp[0], pp[1])

    camera_parameters.intrinsic = intrinsic
    camera_parameters.extrinsic = torch.inverse(camera_to_world).cpu().numpy()

    return camera_parameters
