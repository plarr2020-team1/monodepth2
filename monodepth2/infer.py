import matplotlib as mpl
import matplotlib.cm as cm
from torchvision import transforms, datasets

import monodepth2.networks
from monodepth2.layers import disp_to_depth
from monodepth2.utils import download_model_if_doesnt_exist

import torch
import os
import PIL.Image as pil
import numpy as np


def load_model(model_name):
    """
    Returns an encoder, depth decoder, and expected input image size from a
    model name
    Args:
        model_name: One of:
            "mono_640x192",
            "stereo_640x192",
            "mono+stereo_640x192",
            "mono_no_pt_640x192",
            "stereo_no_pt_640x192",
            "mono+stereo_no_pt_640x192",
            "mono_1024x320",
            "stereo_1024x320",
            "mono+stereo_1024x320"
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = monodepth2.networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if
                        k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = monodepth2.networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, (feed_width, feed_height)


def infer_depth(encoder, depth_decoder, input_size, img):
    """
    Infer depth on an image
    Args:
        encoder: Model encoder
        depth_decoder: Model depth decoder.
        input_size: Input size the model was trained on
        img: Pillow image
    Returns: depth_array, disparity_image
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load image and preprocess
    input_image = img.convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize(input_size, pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear",
        align_corners=False)

    _, scaled_depth = disp_to_depth(disp_resized, 0.1, 100)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(
        np.uint8)
    disp_img = pil.fromarray(colormapped_im)

    return scaled_depth.cpu().detach().numpy(), disp_img
