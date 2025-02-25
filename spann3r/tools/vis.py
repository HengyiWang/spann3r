import os
import cv2
import imageio
import numpy as np
import open3d as o3d
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import open3d as o3d
import os
import numpy as np
import imageio
import os.path as osp

def render_frames(pts_all, image_all, camera_parameters, output_dir, mask=None, save_video=True, save_camera=True, dynamic=False):
    t, h, w, _ = pts_all.shape

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)

    render_frame_path = os.path.join(output_dir, 'render_frames')
    os.makedirs(render_frame_path, exist_ok=True)

    if save_camera:
        o3d.io.write_pinhole_camera_parameters(os.path.join(render_frame_path, 'camera.json'), camera_parameters)

    video_path = os.path.join(output_dir, 'render_frame.mp4')
    if save_video:
        writer = imageio.get_writer(video_path, fps=10)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    for i in range(t):
        new_pts = pts_all[i].reshape(-1, 3)
        new_colors = image_all[i].reshape(-1, 3)

        if mask is not None:
            new_pts = new_pts[mask[i].reshape(-1)]
            new_colors = new_colors[mask[i].reshape(-1)]

        if dynamic:
            pcd.points = o3d.utility.Vector3dVector(new_pts)
            pcd.colors = o3d.utility.Vector3dVector(new_colors)
        else:
            pcd.points.extend(o3d.utility.Vector3dVector(new_pts))
            pcd.colors.extend(o3d.utility.Vector3dVector(new_colors))

        vis.clear_geometries()
        vis.add_geometry(pcd)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_parameters)

        opt = vis.get_render_option()
        opt.point_size = 1
        opt.background_color = np.array([0, 0, 0])

        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(do_render=True)
        image_uint8 = (np.asarray(image) * 255).astype(np.uint8)

        frame_filename = f'frame_{i:03d}.png'
        imageio.imwrite(osp.join(render_frame_path, frame_filename), image_uint8)

        if save_video:
            writer.append_data(image_uint8)

    if save_video:
        writer.close()

    vis.destroy_window()

def draw_camera(c2w, cam_width=0.32/2, cam_height=0.24/2, f=0.10, color=[0, 1, 0], show_axis=True):
    points = [[0, 0, 0], [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
              [cam_width, cam_height, f], [-cam_width, cam_height, f]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(c2w)

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis.scale(min(cam_width, cam_height), np.array([0., 0., 0.]))
        axis.transform(c2w)
        return [line_set, axis]
    else:
        return [line_set]

def find_render_cam(pcd, poses_all=None, cam_width=0.016, cam_height=0.012, cam_f=0.02):
    last_camera_params = None

    def print_camera_pose(vis):
        nonlocal last_camera_params
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        last_camera_params = camera_params 
        
        print("Intrinsic matrix:")
        print(camera_params.intrinsic.intrinsic_matrix)
        print("\nExtrinsic matrix:")
        print(camera_params.extrinsic)
        
        return False
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1920, height=1080)
    vis.add_geometry(pcd)
    if poses_all is not None:
        for pose in poses_all:
            for geometry in draw_camera(pose, cam_width, cam_height, cam_f):
                vis.add_geometry(geometry)

    opt = vis.get_render_option()
    opt.point_size = 1
    opt.background_color = np.array([0, 0, 0])

    vis.register_key_callback(32, print_camera_pose)

    while vis.poll_events():
        vis.update_renderer()

    vis.destroy_window()

    return last_camera_params

def vis_pred_and_imgs(pts_all, save_path, images_all=None, conf_all=None, save_video=True):

    # Normalization
    min_val = pts_all.min(axis=(0, 1, 2), keepdims=True)
    max_val = pts_all.max(axis=(0, 1, 2), keepdims=True)
    pts_all = (pts_all - min_val) / (max_val - min_val)


    pts_save_path = osp.join(save_path, 'pts')
    os.makedirs(pts_save_path, exist_ok=True)

    if images_all is not None:
        images_save_path = osp.join(save_path, 'imgs')
        os.makedirs(images_save_path, exist_ok=True)
    
    if conf_all is not None:
        conf_save_path = osp.join(save_path, 'confs')
        os.makedirs(conf_save_path, exist_ok=True)

    if save_video:
        pts_video_path = osp.join(save_path, 'pts.mp4')
        pts_writer = imageio.get_writer(pts_video_path, fps=10)

        if images_all is not None:
            imgs_video_path = osp.join(save_path, 'imgs.mp4')
            imgs_writer = imageio.get_writer(imgs_video_path, fps=10)

        if conf_all is not None:
            conf_video_path = osp.join(save_path, 'confs.mp4')
            conf_writer = imageio.get_writer(conf_video_path, fps=10)
        
    for frame_id in range(pts_all.shape[0]):
        pt_vis = pts_all[frame_id].astype(np.float32)
        pt_vis_rgb = mcolors.hsv_to_rgb(1-pt_vis)
        pt_vis_rgb_uint8 = (pt_vis_rgb * 255).astype(np.uint8)

        plt.imsave(osp.join(pts_save_path, f'pts_{frame_id:04d}.png'), pt_vis_rgb_uint8)

        if save_video:
            pts_writer.append_data(pt_vis_rgb_uint8)

        

        if images_all is not None:
            image = images_all[frame_id]
            image_uint8 = (image * 255).astype(np.uint8)

            imageio.imwrite(osp.join(images_save_path, f'img_{frame_id:04d}.png'), image_uint8)

            if save_video:
                imgs_writer.append_data(image_uint8)
        
        if conf_all is not None:
            conf_image = plt.cm.jet(conf_all[frame_id])
            conf_image_uint8 = (conf_image * 255).astype(np.uint8)

            plt.imsave(osp.join(conf_save_path, f'conf_{frame_id:04d}.png'), conf_image_uint8)

            if save_video:
                conf_writer.append_data(conf_image_uint8)
    
    if save_video:
        pts_writer.close()
        if images_all is not None:
            imgs_writer.close()
        if conf_all is not None:
            conf_writer.close()





    


