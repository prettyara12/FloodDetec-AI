import pytest
import os
import numpy as np

def test_build_attention_unet_structure():
    from backend.model_builder import build_attention_unet_layers
    layers_dict = build_attention_unet_layers()
    assert "conv2d_30" in layers_dict
    assert "Encoder1" in layers_dict
    assert "Attention1" in layers_dict
    assert "Decoder1" in layers_dict

def test_h5_weights_can_be_mapped():
    from backend.model_builder import extract_weights_from_h5
    h5_path = r"D:\Mata Kuliah\Artificial Intellegence\model_training.h5"
    if os.path.exists(h5_path):
        weights = extract_weights_from_h5(h5_path)
        assert len(weights) > 50
        assert "Encoder1/conv2d/kernel:0" in weights
        assert "Attention1/conv2d_10/kernel:0" in weights
        assert "conv2d_30/kernel:0" in weights
