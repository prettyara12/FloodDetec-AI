"""
Attention U-Net Model Builder & Weights Loader
Reverse-engineered for Flood Area Segmentation from model_training.h5.
"""

from typing import Dict, Any, Tuple
import os
import h5py
import numpy as np


def extract_weights_from_h5(h5_path: str) -> Dict[str, np.ndarray]:
    """
    Extracts all weights directly from the HDF5 model weights group.
    Returns a dictionary mapping full tensor name -> numpy ndarray.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model file not found at: {h5_path}")

    weights_dict: Dict[str, np.ndarray] = {}

    with h5py.File(h5_path, "r") as f:
        if "model_weights" not in f:
            raise ValueError(f"HDF5 file does not contain 'model_weights' root key: {h5_path}")

        weights_group = f["model_weights"]

        def visitor(name: str, node: Any):
            if isinstance(node, h5py.Dataset):
                # node.name is e.g. /model_weights/Encoder1/Encoder1/conv2d/kernel:0
                # Extract relative path inside model_weights
                parts = name.split("/")
                # usually parts are [LayerName, SubLayerName, weight_name]
                clean_name = "/".join(parts[-3:]) if len(parts) >= 3 else name
                weights_dict[clean_name] = np.array(node)
                # also map by basename for easy lookup
                simple_name = "/".join(parts[-2:])
                weights_dict[simple_name] = np.array(node)

        weights_group.visititems(visitor)

    return weights_dict


def build_attention_unet_layers() -> Dict[str, Any]:
    """
    Defines the exact structural schema and dimensions of Attention U-Net
    extracted from model_training.h5.
    """
    return {
        "input_shape": (256, 256, 3),
        "output_shape": (256, 256, 1),
        "Encoder1": {
            "filters": 32,
            "layers": ["conv2d", "conv2d_1"],
            "pooling": True,
            "kernel_shapes": {"conv2d": (3, 3, 3, 32), "conv2d_1": (3, 3, 32, 32)}
        },
        "Encoder2": {
            "filters": 64,
            "layers": ["conv2d_2", "conv2d_3"],
            "pooling": True,
            "kernel_shapes": {"conv2d_2": (3, 3, 32, 64), "conv2d_3": (3, 3, 64, 64)}
        },
        "Encoder3": {
            "filters": 128,
            "layers": ["conv2d_4", "conv2d_5"],
            "pooling": True,
            "kernel_shapes": {"conv2d_4": (3, 3, 64, 128), "conv2d_5": (3, 3, 128, 128)}
        },
        "Encoder4": {
            "filters": 256,
            "layers": ["conv2d_6", "conv2d_7"],
            "pooling": True,
            "kernel_shapes": {"conv2d_6": (3, 3, 128, 256), "conv2d_7": (3, 3, 256, 256)}
        },
        "Encoding": {
            "filters": 512,
            "layers": ["conv2d_8", "conv2d_9"],
            "pooling": False,
            "kernel_shapes": {"conv2d_8": (3, 3, 256, 512), "conv2d_9": (3, 3, 512, 512)}
        },
        "Attention1": {
            "filters": 256,
            "gating_shape": (32, 32, 512),
            "skip_shape": (32, 32, 256),
            "layers": ["conv2d_10", "conv2d_11", "batch_normalization", "conv2d_12"]
        },
        "Decoder1": {
            "filters": 256,
            "in_channels": 768,
            "layers": ["conv2d_13", "conv2d_14"]
        },
        "Attention2": {
            "filters": 128,
            "gating_shape": (64, 64, 256),
            "skip_shape": (64, 64, 128),
            "layers": ["conv2d_15", "conv2d_16", "batch_normalization_1", "conv2d_17"]
        },
        "Decoder2": {
            "filters": 128,
            "in_channels": 384,
            "layers": ["conv2d_18", "conv2d_19"]
        },
        "Attention3": {
            "filters": 64,
            "gating_shape": (128, 128, 128),
            "skip_shape": (128, 128, 64),
            "layers": ["conv2d_20", "conv2d_21", "batch_normalization_2", "conv2d_22"]
        },
        "Decoder3": {
            "filters": 64,
            "in_channels": 192,
            "layers": ["conv2d_23", "conv2d_24"]
        },
        "Attention4": {
            "filters": 32,
            "gating_shape": (256, 256, 64),
            "skip_shape": (256, 256, 32),
            "layers": ["conv2d_25", "conv2d_26", "batch_normalization_3", "conv2d_27"]
        },
        "Decoder4": {
            "filters": 32,
            "in_channels": 96,
            "layers": ["conv2d_28", "conv2d_29"]
        },
        "conv2d_30": {
            "filters": 1,
            "kernel_size": (1, 1),
            "activation": "sigmoid"
        }
    }
