import os
import cv2
import copy
import trimesh
import pyrender
import numpy as np
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt

def load_cam_mvsnet(file, interval_scale=1):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 192
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0
        
        
        extrinsic = cam[0].astype(np.float32)
        intrinsic = cam[1].astype(np.float32)

        return intrinsic, extrinsic

def render_depth_maps(mesh, poses, K, H, W, near=0.01, far=5.0):
    """
    :param mesh: Mesh to be rendered
    :param poses: list of camera poses (c2w under OpenGL convention)
    :param K: camera intrinsics [3, 3]
    :param W: width of image plane
    :param H: height of image plane
    :param near: near clip
    :param far: far clip
    :return: list of rendered depth images [H, W]
    """
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=near, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

    depth_maps = []
    for pose in poses:
        scene.set_pose(camera_node, pose)
        depth = renderer.render(scene, render_flags)
        depth_maps.append(depth)

    return depth_maps

def render_dtu_scenes(path_to_scan, method='furu'):
    path_to_cameras = os.path.join(path_to_scan, 'cams')
    path_to_images = os.path.join(path_to_scan, 'images')
    scan_id = int(''.join(filter(str.isdigit, os.path.basename(path_to_scan))))
    if method is not None:
        path_to_depths = os.path.join(path_to_scan, f'depths_{method}')
        path_to_mesh = os.path.join(path_to_scan, f'{method}{scan_id:03d}_l3_surf_11_trim_8.ply')
    else:
        path_to_depths = os.path.join(path_to_scan, 'depths')
        path_to_mesh = os.path.join(path_to_scan, f'{scan_id:03d}_pcd.ply')
    
    #path_to_mesh = os.path.join(path_to_scan, 'stl001_total.ply')
    
    if not os.path.exists(path_to_depths):
        os.makedirs(path_to_depths)
    
    mesh = trimesh.load_mesh(path_to_mesh)
    
    frames = sorted(os.listdir(path_to_images))
    
    img = cv2.imread(os.path.join(path_to_images, frames[0]))
    H, W, _ = img.shape
    
    for i, frame in enumerate(frames):
        campath = os.path.join(path_to_cameras, frame.replace('.jpg', '_cam.txt'))
        print(campath)
        cur_intrinsics, camera_pose = load_cam_mvsnet(open(campath, 'r'))
        camera_pose = np.linalg.inv(camera_pose)
        
        camera_pose[:, 1:3] *= -1.0
        
        
        
        print(cur_intrinsics)
        
        depth = render_depth_maps(mesh, [camera_pose], cur_intrinsics, H, W, near=0.01, far=5000.)[0]
        
        # plt.imshow(depth)
        # plt.show()      
        # Save depth map
        #cv2.imwrite(os.path.join(path_to_depths, frame.replace('.jpg', '.png')), depth)
        # depth_16bit = (depth).astype(np.uint16)  # Scale to millimeters
        
        # # Save depth map as 16-bit PNG
        # depth_filename = os.path.join(path_to_depths, frame.replace('.jpg', '.png'))
        # cv2.imwrite(depth_filename, depth_16bit)
        # depth = depth.astype(np.float32)
        depth_filename = os.path.join(path_to_depths, frame.replace('.jpg', '.npy'))
        np.save(depth_filename, depth)
        
def get_dtu_mask(path_to_scan, method='furu'):
    
    if method is not None:
        path_to_depths = os.path.join(path_to_scan, f'depths_{method}')
        path_to_masks = os.path.join(path_to_scan, f'masks_{method}')
    
    else:
        path_to_depths = os.path.join(path_to_scan, 'depths')
        path_to_masks = os.path.join(path_to_scan, 'masks')
    
    if not os.path.exists(path_to_masks):
        os.makedirs(path_to_masks)
    

    frames = sorted(os.listdir(path_to_depths))
    
    for i, frame in enumerate(frames):
        depth_filename = os.path.join(path_to_depths, frame)        
        depth = np.load(depth_filename)
        
        mask_path = os.path.join(path_to_masks, frame.replace('.npy', '.png'))
        
        mask = np.ones_like(depth) * 255
        mask[depth == 0] = 0
        mask[depth>900] = 0
        cv2.imwrite(mask_path, mask)


def get_mesh_from_ply(path_to_scan, depth=9, density_thresh=0.1):
    scan_id = int(''.join(filter(str.isdigit, os.path.basename(path_to_scan))))
    path_to_ply = os.path.join(path_to_scan, f'stl{scan_id:03d}_total.ply')
    
    pcd = o3d.io.read_point_cloud(path_to_ply)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    
    vertices_to_remove = densities < np.quantile(densities, density_thresh)

    new_mesh = copy.deepcopy(mesh)      
    new_mesh = copy.deepcopy(mesh)
    new_mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # save mesh
    path_to_mesh = os.path.join(path_to_scan, f'{scan_id:03d}_pcd.ply')
    o3d.io.write_triangle_mesh(path_to_mesh, new_mesh)
    





path_to_dtu = './data/dtu_test'

scans = sorted(os.listdir(path_to_dtu))

for scan in tqdm(scans):
    print(f"Processing {scan}")
    
    path_to_scan = os.path.join(path_to_dtu, scan)
    
    get_mesh_from_ply(path_to_scan)
    
    render_dtu_scenes(path_to_scan, method=None)
    #get_dtu_mask(path_to_scan, None)
    
