import os
import time
import torch
import numpy as np
import gradio as gr
import urllib.parse
import tempfile
import subprocess
from dust3r.losses import L21
from spann3r.model import Spann3R
from spann3r.datasets import Demo
from torch.utils.data import DataLoader
import trimesh
from scipy.spatial.transform import Rotation
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image

# Default values
DEFAULT_CKPT_PATH = 'https://huggingface.co/spaces/Stable-X/StableSpann3R/resolve/main/checkpoints/spann3r.pth'
DEFAULT_DUST3R_PATH = 'https://huggingface.co/camenduru/dust3r/resolve/main/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

def extract_frames(video_path: str) -> str:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "%03d.jpg")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "fps=1",
        output_path
    ]
    subprocess.run(command, check=True)
    return temp_dir

def cat_meshes(meshes):
    vertices, faces, colors = zip(*[(m['vertices'], m['faces'], m['face_colors']) for m in meshes])
    n_vertices = np.cumsum([0]+[len(v) for v in vertices])
    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    vertices = np.concatenate(vertices)
    colors = np.concatenate(colors)
    faces = np.concatenate(faces)
    return dict(vertices=vertices, face_colors=colors, faces=faces)

def load_ckpt(model_path_or_url, verbose=True):
    if verbose:
        print('... loading model from', model_path_or_url)
    is_url = urllib.parse.urlparse(model_path_or_url).scheme in ('http', 'https')
    
    if is_url:
        ckpt = torch.hub.load_state_dict_from_url(model_path_or_url, map_location='cpu', progress=verbose)
    else:
        ckpt = torch.load(model_path_or_url, map_location='cpu')
    return ckpt

def load_model(ckpt_path, device):
    model = Spann3R(dus3r_name=DEFAULT_DUST3R_PATH, 
                    use_feat=False).to(device)
    
    model.load_state_dict(load_ckpt(ckpt_path)['model'])
    model.eval()
    return model

def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate((
        np.c_[idx1, idx2, idx3],
        np.c_[idx3, idx2, idx1],  # same triangle, but backward (cheap solution to cancel face culling)
        np.c_[idx2, idx3, idx4],
        np.c_[idx4, idx3, idx2],  # same triangle, but backward (cheap solution to cancel face culling)
    ), axis=0)

    # prepare triangle colors
    face_colors = np.concatenate((
        img[:-1, :-1].reshape(-1, 3),
        img[:-1, :-1].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3)
    ), axis=0)

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)

model = load_model(DEFAULT_CKPT_PATH, DEFAULT_DEVICE)
birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
birefnet.to(DEFAULT_DEVICE)
birefnet.eval()

def extract_object(birefnet, image):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_images = transform_image(image).unsqueeze(0).to(DEFAULT_DEVICE)

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    return mask

def generate_mask(image: np.ndarray):
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    
    # Extract object and get mask
    mask = extract_object(birefnet, pil_image)
    
    # Convert mask to numpy array
    mask_np = np.array(mask) / 255.0
    return mask_np

@torch.no_grad()
def reconstruct(video_path, conf_thresh, kf_every, as_pointcloud=False, remove_background=False):
    # Extract frames from video
    demo_path = extract_frames(video_path)
    
    # Load dataset
    dataset = Demo(ROOT=demo_path, resolution=224, full_video=True, kf_every=kf_every)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    
    for view in batch:
        view['img'] = view['img'].to(DEFAULT_DEVICE, non_blocking=True)
    
    demo_name = os.path.basename(video_path)
    print(f'Started reconstruction for {demo_name}')
    
    start = time.time()
    preds, preds_all = model.forward(batch)
    end = time.time()
    fps = len(batch) / (end - start)
    print(f'Finished reconstruction for {demo_name}, FPS: {fps:.2f}')
    
    # Process results
    pts_all, images_all, conf_all, mask_all = [], [], [], []
    for j, view in enumerate(batch):
        image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
        pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
        conf = preds[j]['conf'][0].cpu().data.numpy()
        
        if remove_background:
            mask = generate_mask(image)
        else:
            mask = np.ones_like(conf)  # Change this to match conf shape
        
        images_all.append((image[None, ...] + 1.0)/2.0)
        pts_all.append(pts[None, ...])
        conf_all.append(conf[None, ...])
        mask_all.append(mask[None, ...])
    
    images_all = np.concatenate(images_all, axis=0)
    pts_all = np.concatenate(pts_all, axis=0) * 10
    conf_all = np.concatenate(conf_all, axis=0)
    mask_all = np.concatenate(mask_all, axis=0)
    
    # Create point cloud or mesh
    conf_sig_all = (conf_all-1) / conf_all
    combined_mask = (conf_sig_all > conf_thresh) & (mask_all > 0.5)
    
    scene = trimesh.Scene()
    
    if as_pointcloud:
        pcd = trimesh.PointCloud(
            vertices=pts_all[combined_mask].reshape(-1, 3),
            colors=images_all[combined_mask].reshape(-1, 3)
        )
        scene.add_geometry(pcd)
    else:
        meshes = []
        for i in range(len(images_all)):
            meshes.append(pts3d_to_trimesh(images_all[i], pts_all[i], combined_mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)
    
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(OPENGL @ rot))
    
    # Save the scene as GLB
    if as_pointcloud:
        output_path = tempfile.mktemp(suffix='.ply')  
    else:
        output_path = tempfile.mktemp(suffix='.obj')
    scene.export(output_path)
    
    # Clean up temporary directory
    os.system(f"rm -rf {demo_path}")
    
    return output_path, f"Reconstruction completed. FPS: {fps:.2f}"

iface = gr.Interface(
    fn=reconstruct,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Slider(0, 1, value=1e-3, label="Confidence Threshold"),
        gr.Slider(1, 30, step=1, value=5, label="Keyframe Interval"),
        gr.Checkbox(label="As Pointcloud", value=False),
        gr.Checkbox(label="Remove Background", value=False)
    ],
    outputs=[
        gr.Model3D(label="3D Model", display_mode="solid"),
        gr.Textbox(label="Status")
    ],
    title="3D Reconstruction with Spatial Memory and Background Removal",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0",)