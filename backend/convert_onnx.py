"""
ONNX Model Converter for Attention U-Net
Converts Keras model weights from model_training.h5 to an optimized ONNX model graph.
"""

import os
from typing import Dict, Any, List
import h5py
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def export_h5_to_onnx(h5_path: str, output_onnx_path: str = "models/model.onnx") -> str:
    """
    Reads weights from h5_path, constructs an Attention U-Net ONNX graph,
    and writes the model to output_onnx_path.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Source model file not found: {h5_path}")

    # Extract all weights
    weights: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for layer_name in f["model_weights"]:
            grp = f["model_weights"][layer_name]
            def visitor(name: str, node: Any):
                if isinstance(node, h5py.Dataset):
                    parts = name.split("/")
                    key = "/".join(parts[-2:])
                    weights[key] = np.array(node)
                    weights[name] = np.array(node)
            grp.visititems(visitor)

    nodes: List[Any] = []
    initializers: List[Any] = []

    def add_initializer(name: str, arr: np.ndarray) -> str:
        tensor = numpy_helper.from_array(arr.astype(np.float32), name=name)
        initializers.append(tensor)
        return name

    def make_conv(input_name: str, conv_prefix: str, out_name: str, activation: str = None) -> str:
        w_k = weights[f"{conv_prefix}/kernel:0"]
        w_b = weights[f"{conv_prefix}/bias:0"]
        # Transpose Keras (kH, kW, inC, outC) to ONNX (outC, inC, kH, kW)
        w_k_onnx = np.transpose(w_k, (3, 2, 0, 1))
        w_name = add_initializer(f"{conv_prefix}_w", w_k_onnx)
        b_name = add_initializer(f"{conv_prefix}_b", w_b)

        pad = 1 if w_k.shape[0] == 3 else 0
        conv_out = f"{out_name}_conv" if activation else out_name
        conv_node = helper.make_node(
            "Conv",
            inputs=[input_name, w_name, b_name],
            outputs=[conv_out],
            kernel_shape=[w_k.shape[0], w_k.shape[1]],
            pads=[pad, pad, pad, pad],
            strides=[1, 1],
            name=f"node_{conv_prefix}"
        )
        nodes.append(conv_node)

        if activation == "relu":
            nodes.append(helper.make_node("Relu", inputs=[conv_out], outputs=[out_name], name=f"relu_{out_name}"))
        elif activation == "sigmoid":
            nodes.append(helper.make_node("Sigmoid", inputs=[conv_out], outputs=[out_name], name=f"sig_{out_name}"))
        return out_name

    def make_maxpool(input_name: str, out_name: str) -> str:
        pool_node = helper.make_node(
            "MaxPool",
            inputs=[input_name],
            outputs=[out_name],
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
            name=f"pool_{out_name}"
        )
        nodes.append(pool_node)
        return out_name

    def make_upsample2x(input_name: str, out_name: str) -> str:
        roi = add_initializer(f"roi_{out_name}", np.array([], dtype=np.float32))
        scales = add_initializer(f"scales_{out_name}", np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
        resize_node = helper.make_node(
            "Resize",
            inputs=[input_name, roi, scales],
            outputs=[out_name],
            mode="nearest",
            name=f"resize_{out_name}"
        )
        nodes.append(resize_node)
        return out_name

    def make_batchnorm(input_name: str, bn_prefix: str, out_name: str) -> str:
        gamma = add_initializer(f"{bn_prefix}_gamma", weights[f"{bn_prefix}/gamma:0"])
        beta = add_initializer(f"{bn_prefix}_beta", weights[f"{bn_prefix}/beta:0"])
        mean = add_initializer(f"{bn_prefix}_mean", weights[f"{bn_prefix}/moving_mean:0"])
        var = add_initializer(f"{bn_prefix}_var", weights[f"{bn_prefix}/moving_variance:0"])
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=[input_name, gamma, beta, mean, var],
            outputs=[out_name],
            epsilon=1e-5,
            name=f"bn_{out_name}"
        )
        nodes.append(bn_node)
        return out_name

    # Input: input_1 shape [1, 256, 256, 3] (NHWC) -> Transpose to NCHW: [1, 3, 256, 256]
    nodes.append(helper.make_node("Transpose", inputs=["input_1"], outputs=["x_nchw"], perm=[0, 3, 1, 2], name="trans_in"))

    # Encoder 1
    c1 = make_conv("x_nchw", "conv2d", "c1", "relu")
    enc1_skip = make_conv(c1, "conv2d_1", "enc1_skip", "relu")
    enc1_pool = make_maxpool(enc1_skip, "enc1_pool")

    # Encoder 2
    c2 = make_conv(enc1_pool, "conv2d_2", "c2", "relu")
    enc2_skip = make_conv(c2, "conv2d_3", "enc2_skip", "relu")
    enc2_pool = make_maxpool(enc2_skip, "enc2_pool")

    # Encoder 3
    c3 = make_conv(enc2_pool, "conv2d_4", "c3", "relu")
    enc3_skip = make_conv(c3, "conv2d_5", "enc3_skip", "relu")
    enc3_pool = make_maxpool(enc3_skip, "enc3_pool")

    # Encoder 4
    c4 = make_conv(enc3_pool, "conv2d_6", "c4", "relu")
    enc4_skip = make_conv(c4, "conv2d_7", "enc4_skip", "relu")
    enc4_pool = make_maxpool(enc4_skip, "enc4_pool")

    # Encoding (Bottleneck)
    c_b1 = make_conv(enc4_pool, "conv2d_8", "c_b1", "relu")
    encoding = make_conv(c_b1, "conv2d_9", "encoding", "relu")

    # Attention Gate 1
    wg1 = make_conv(encoding, "conv2d_10", "wg1")
    wg1_up = make_upsample2x(wg1, "wg1_up")
    ws1 = make_conv(enc4_skip, "conv2d_11", "ws1")
    nodes.append(helper.make_node("Add", inputs=[wg1_up, ws1], outputs=["att1_add"], name="add_att1"))
    bn1 = make_batchnorm("att1_add", "batch_normalization", "att1_bn")
    nodes.append(helper.make_node("Relu", inputs=[bn1], outputs=["att1_relu"], name="relu_att1"))
    psi1 = make_conv("att1_relu", "conv2d_12", "psi1", "sigmoid")
    nodes.append(helper.make_node("Mul", inputs=[enc4_skip, psi1], outputs=["att1_out"], name="mul_att1"))

    # Decoder 1
    enc_up1 = make_upsample2x(encoding, "enc_up1")
    nodes.append(helper.make_node("Concat", inputs=[enc_up1, "att1_out"], outputs=["concat1"], axis=1, name="cat1"))
    d1_1 = make_conv("concat1", "conv2d_13", "d1_1", "relu")
    dec1 = make_conv(d1_1, "conv2d_14", "dec1", "relu")

    # Attention Gate 2
    wg2 = make_conv(dec1, "conv2d_15", "wg2")
    wg2_up = make_upsample2x(wg2, "wg2_up")
    ws2 = make_conv(enc3_skip, "conv2d_16", "ws2")
    nodes.append(helper.make_node("Add", inputs=[wg2_up, ws2], outputs=["att2_add"], name="add_att2"))
    bn2 = make_batchnorm("att2_add", "batch_normalization_1", "att2_bn")
    nodes.append(helper.make_node("Relu", inputs=[bn2], outputs=["att2_relu"], name="relu_att2"))
    psi2 = make_conv("att2_relu", "conv2d_17", "psi2", "sigmoid")
    nodes.append(helper.make_node("Mul", inputs=[enc3_skip, psi2], outputs=["att2_out"], name="mul_att2"))

    # Decoder 2
    dec1_up = make_upsample2x(dec1, "dec1_up")
    nodes.append(helper.make_node("Concat", inputs=[dec1_up, "att2_out"], outputs=["concat2"], axis=1, name="cat2"))
    d2_1 = make_conv("concat2", "conv2d_18", "d2_1", "relu")
    dec2 = make_conv(d2_1, "conv2d_19", "dec2", "relu")

    # Attention Gate 3
    wg3 = make_conv(dec2, "conv2d_20", "wg3")
    wg3_up = make_upsample2x(wg3, "wg3_up")
    ws3 = make_conv(enc2_skip, "conv2d_21", "ws3")
    nodes.append(helper.make_node("Add", inputs=[wg3_up, ws3], outputs=["att3_add"], name="add_att3"))
    bn3 = make_batchnorm("att3_add", "batch_normalization_2", "att3_bn")
    nodes.append(helper.make_node("Relu", inputs=[bn3], outputs=["att3_relu"], name="relu_att3"))
    psi3 = make_conv("att3_relu", "conv2d_22", "psi3", "sigmoid")
    nodes.append(helper.make_node("Mul", inputs=[enc2_skip, psi3], outputs=["att3_out"], name="mul_att3"))

    # Decoder 3
    dec2_up = make_upsample2x(dec2, "dec2_up")
    nodes.append(helper.make_node("Concat", inputs=[dec2_up, "att3_out"], outputs=["concat3"], axis=1, name="cat3"))
    d3_1 = make_conv("concat3", "conv2d_23", "d3_1", "relu")
    dec3 = make_conv(d3_1, "conv2d_24", "dec3", "relu")

    # Attention Gate 4
    wg4 = make_conv(dec3, "conv2d_25", "wg4")
    wg4_up = make_upsample2x(wg4, "wg4_up")
    ws4 = make_conv(enc1_skip, "conv2d_26", "ws4")
    nodes.append(helper.make_node("Add", inputs=[wg4_up, ws4], outputs=["att4_add"], name="add_att4"))
    bn4 = make_batchnorm("att4_add", "batch_normalization_3", "att4_bn")
    nodes.append(helper.make_node("Relu", inputs=[bn4], outputs=["att4_relu"], name="relu_att4"))
    psi4 = make_conv("att4_relu", "conv2d_27", "psi4", "sigmoid")
    nodes.append(helper.make_node("Mul", inputs=[enc1_skip, psi4], outputs=["att4_out"], name="mul_att4"))

    # Decoder 4
    dec3_up = make_upsample2x(dec3, "dec3_up")
    nodes.append(helper.make_node("Concat", inputs=[dec3_up, "att4_out"], outputs=["concat4"], axis=1, name="cat4"))
    d4_1 = make_conv("concat4", "conv2d_28", "d4_1", "relu")
    dec4 = make_conv(d4_1, "conv2d_29", "dec4", "relu")

    # Output Layer
    conv30 = make_conv(dec4, "conv2d_30", "conv30_nchw", "sigmoid")
    nodes.append(helper.make_node("Transpose", inputs=["conv30_nchw"], outputs=["output_1"], perm=[0, 2, 3, 1], name="trans_out"))

    input_tensor_def = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [1, 256, 256, 3])
    output_tensor_def = helper.make_tensor_value_info("output_1", TensorProto.FLOAT, [1, 256, 256, 1])

    graph_def = helper.make_graph(
        nodes,
        "AttentionUNet_FloodSegmentation",
        [input_tensor_def],
        [output_tensor_def],
        initializers
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="geoflood_builder",
        opset_imports=[helper.make_opsetid("", 14)],
        ir_version=10
    )
    onnx.checker.check_model(model_def)

    os.makedirs(os.path.dirname(output_onnx_path) or ".", exist_ok=True)
    onnx.save(model_def, output_onnx_path)
    return output_onnx_path


if __name__ == "__main__":
    h5_default = r"models/model_training.h5"
    if not os.path.exists(h5_default):
        h5_default = r"D:\Mata Kuliah\Artificial Intellegence\model_training.h5"
    out_path = export_h5_to_onnx(h5_default, "models/model.onnx")
    print(f"Export completed: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")
