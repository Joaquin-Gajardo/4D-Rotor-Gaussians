import json
from pathlib import Path
from PIL import Image
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch

from nerfstudio.process_data.colmap_utils import (
    create_ply_from_colmap,
    parse_colmap_camera_params,
)   
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_points3D_text,
)

import os
from pathlib import Path
from PIL import Image
import json
import numpy as np
from typing import Dict, Optional

def generate_image_map(input_dir: Path, output_dir: Path) -> Dict[str, str]:
    """
    Generate a mapping between original and downsampled image paths without performing downsampling.

    Args:
        input_dir: Path to the input directory containing the original images.
        output_dir: Path to the output directory where downsampled images are (or would be) saved.

    Returns:
        A dictionary mapping original image paths to downsampled image paths.
    """
    image_map = {}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = Path(root) / file
                rel_path = input_path.relative_to(input_dir)
                output_path = output_dir / rel_path
                image_map[str(input_path)] = str(output_path)
    return image_map

def downsample_images(input_dir: Path, output_dir: Path, scale_factor: float = 0.5, force: bool = False) -> Dict[str, str]:
    """
    Downsample images in the input directory and save them to the output directory.
    Maintains the original folder structure and file names.

    Args:
        input_dir: Path to the input directory containing the images.
        output_dir: Path to the output directory where downsampled images will be saved.
        scale_factor: Factor by which to scale down the images (default is 0.5 for 2x downsampling).
        force: If True, overwrite existing downsampled images. If False, skip existing images.

    Returns:
        A dictionary mapping original image paths to downsampled image paths.
    """
    image_map = generate_image_map(input_dir, output_dir)
    
    for input_path, output_path in image_map.items():
        output_path = Path(output_path)
        if force or not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(input_path) as img:
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                img_downsampled = img.resize(new_size, Image.LANCZOS)
                img_downsampled.save(output_path)
    
    return image_map

# Based on nerfstudio/process_data/colmap_utils.py
def colmap_to_json(
    base_dir: Path,
    sparse_dir: str = "sparse/0",
    image_dir: str = "images_downsampled",
    scale_factor: float = 0.5,
    ply_filename="sparse_pc.ply",
    keep_original_world_coordinate: bool = False,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        image_map: Dictionary mapping original image paths to downsampled image paths.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        keep_original_world_coordinate: If True, no extra transform will be applied to world coordinate.
                    Colmap optimized world often have y direction of the first camera pointing towards down direction,
                    while nerfstudio world set z direction to be up direction for viewer.
    Returns:
        The number of registered images.
    """

    cam_id_to_camera = read_cameras_binary(base_dir / sparse_dir / "cameras.bin")
    im_id_to_image = read_images_binary(base_dir / sparse_dir / "images.bin")

    subfolders = sorted(set([im.name.split('/')[0] for im in im_id_to_image.values()]))
    timestep_mapping = {subfolder: idx for idx, subfolder in enumerate(subfolders)}

    frames = []
    for im_id, im_data in im_id_to_image.items():
        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        if not keep_original_world_coordinate:
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1

        name = im_data.name
        rel_image_path = Path(f"./{image_dir}/{name}")

        # Calculate time based on the subfolder
        subfolder = rel_image_path.parent.name
        time = timestep_mapping[subfolder]

        frame = {
            "file_path": rel_image_path.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
            "time": time,
        }
        frames.append(frame)

    if set(cam_id_to_camera.keys()) != {1}:
        print("Warning: Only single camera shared for all images is supported. Choosing the first one")
        print(cam_id_to_camera)
    
    out = parse_colmap_camera_params(cam_id_to_camera[1])

    # Adjust camera parameters for downsampling
    out["w"] = int(out["w"] * scale_factor)
    out["h"] = int(out["h"] * scale_factor)
    out["fl_x"] *= scale_factor
    out["fl_y"] *= scale_factor
    out["cx"] *= scale_factor
    out["cy"] *= scale_factor
    # NOTE: k1 (radial distortion) doesn't need rescaling

    # Split into train and test
    sorted_frames = sorted(frames.copy(), key = lambda x : tuple(map(int, Path(x['file_path']).stem.split('_')[1:])))
    llffhold = 8
    train_frames = [c for idx, c in enumerate(sorted_frames) if idx % llffhold != 0]
    test_frames = [c for idx, c in enumerate(sorted_frames) if idx % llffhold == 0]

        # Create train and test outputs
    train_out = out.copy()
    train_out["frames"] = train_frames
    test_out = out.copy()
    test_out["frames"] = test_frames

    applied_transform = None
    if not keep_original_world_coordinate:
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([0, 2, 1]), :]
        applied_transform[2, :] *= -1
        train_out["applied_transform"] = applied_transform.tolist()
        test_out["applied_transform"] = applied_transform.tolist()

    # Write train and test JSONs
    train_json = base_dir / "transforms_train.json"
    test_json = base_dir / "transforms_test.json"
    if not train_json.exists():
        with open(train_json, "w", encoding="utf-8") as f:
            json.dump(train_out, f, indent=4)
    if not test_json.exists():
        with open(test_json, "w", encoding="utf-8") as f:
            json.dump(test_out, f, indent=4)

    return len(frames)

def main():

    scale_factor = 1.0
    input_images_dir = 'images'
    output_images_dir = 'images'
    dataset_path = Path("data/WAT")
    
    for subfolder in sorted(dataset_path.iterdir()):
        if subfolder.is_dir():
            input_dir = subfolder / input_images_dir
            output_dir = subfolder / output_images_dir
            if scale_factor != 1.0:
                print("Downsampling images...")
                downsample_images(input_dir, output_dir, scale_factor=scale_factor)

            if not (subfolder / "transforms_train.json").exists() or not (subfolder / "transforms_test.json").exists():
                print(f"Creating JSON files for scene {subfolder}...")
                colmap_to_json(
                    base_dir = subfolder,
                    image_dir = output_images_dir,
                    scale_factor = scale_factor,
                )
            else:
                print(f"JSON files already exist for scene {subfolder}")

if __name__ == "__main__":

    main()