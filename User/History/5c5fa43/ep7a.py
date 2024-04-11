from collections import OrderedDict
import time
import PIL
import numpy as np
import onnxruntime
import torch
from torchvision import transforms as T
import cv2
from change_detector_model.DeepLabV3 import vgg16bn_mtf_msf_deeplabv3


def predict(model, image1_input, image2_input, device="cpu", input_size=256):
    assert device in ["cpu", "cuda"], "Device must be either 'cpu' or 'cuda'"
    assert input_size > 0, "Input size must be a positive integer"
    start = time.time()
    if isinstance(image1_input, str) and isinstance(image2_input, str):
        # Load image from disk if inputs are paths
        image1 = PIL.Image.open(image1_input).convert("RGB")
        image2 = PIL.Image.open(image2_input).convert("RGB")
    elif isinstance(image1_input, np.ndarray) and isinstance(image2_input, np.ndarray):
        # Convert numpy arrays to PIL Images
        image1 = PIL.Image.fromarray(image1_input.astype("uint8"), "RGB")
        image2 = PIL.Image.fromarray(image2_input.astype("uint8"), "RGB")
        assert image1.size == image2.size, f"Images must have the same size, {image1.size} != {image2.size}"
    else:
        raise ValueError("Inputs must be either paths or numpy arrays")

    print(f"Load time: {time.time() - start:.2f}s")

    original_size = image1.size[::-1]
    print(f"Original size: {original_size}")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    start = time.time()

    augs = []
    augs.append(T.Resize(input_size))
    augs.append(T.ToTensor())
    augs.append(T.Normalize(mean=mean, std=std))
    transform = T.Compose(augs)

    # Convert and preprocess images
    image1 = transform(image1)
    image2 = transform(image2)
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)
    print(f"Preprocess time: {time.time() - start:.2f}s")

    model.eval()
    with torch.no_grad():
        start = time.time()
        image1, image2 = image1.to(device), image2.to(device)
        print(f"Send to device time: {time.time() - start:.2f}s")
        # concatenate in the channel dimension
        image = torch.cat((image1, image2), dim=1)
        print(f"Inference shape: {image.shape}")
        start = time.time()
        output = model(image)
        print(f"Inference time: {time.time() - start:.2f}s")
        if isinstance(output, OrderedDict):
            output = output["out"]
        start = time.time()
        mask_pred = torch.topk(output.data, 1, dim=1)[1][:, 0]
        mask_pred = torch.nn.functional.interpolate(mask_pred.unsqueeze(0).float(), size=original_size, mode="nearest").byte().squeeze(0)
        mask_pred = mask_pred.cpu().numpy().squeeze().astype(np.uint8) * 255
        print(f"Postprocess time: {time.time() - start:.2f}s")
        return mask_pred


def onnx_predict(onnx_model, image1_input, image2_input, input_size=256):
    assert input_size > 0, "Input size must be a positive integer"
    start = time.time()
    if isinstance(image1_input, str) and isinstance(image2_input, str):
        # Load image from disk if inputs are paths
        image1 = PIL.Image.open(image1_input).convert("RGB")
        image2 = PIL.Image.open(image2_input).convert("RGB")
    elif isinstance(image1_input, np.ndarray) and isinstance(image2_input, np.ndarray):
        # Convert numpy arrays to PIL Images
        image1 = PIL.Image.fromarray(image1_input.astype("uint8"), "RGB")
        image2 = PIL.Image.fromarray(image2_input.astype("uint8"), "RGB")
        assert image1.size == image2.size, f"Images must have the same size, {image1.size} != {image2.size}"
    else:
        raise ValueError("Inputs must be either paths or numpy arrays")

    print(f"Load time: {time.time() - start:.2f}s")

    original_size = image1.size[::-1]
    print(f"Original size: {original_size}")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    start = time.time()

    augs = []
    augs.append(T.Resize(input_size))
    augs.append(T.ToTensor())
    augs.append(T.Normalize(mean=mean, std=std))
    transform = T.Compose(augs)

    # Convert and preprocess images
    image1 = transform(image1)
    image2 = transform(image2)
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)
    print(f"Preprocess time: {time.time() - start:.2f}s")

    image = torch.cat((image1, image2), dim=1)

    so = onnxruntime.SessionOptions()
    so.log_severity_level = 0
    ort_session = onnxruntime.InferenceSession("cd.onnx", so)
    print("Model input shape:", ort_session.get_inputs()[0].shape)
    ort_inputs = {ort_session.get_inputs()[0].name: image.cpu().numpy()}
    print("Input shape:", ort_inputs[ort_session.get_inputs()[0].name].shape)
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs_np = ort_outs[0]
    output_tensor = torch.from_numpy(ort_outs_np)
    print("Output shape:", output_tensor.shape)

    mask_pred = torch.topk(output_tensor.data, 1, dim=1)[1][:, 0]
    print("Mask shape:", mask_pred.shape)

    mask_pred = torch.nn.functional.interpolate(mask_pred.unsqueeze(0).float(), size=original_size, mode="nearest").byte().squeeze(0)
    mask_pred = mask_pred.cpu().numpy().squeeze().astype(np.uint8) * 255

    return mask_pred


def load_model(checkpoint_path, device="cpu", input_size=(256)):
    device = torch.device(device)

    model = vgg16bn_mtf_msf_deeplabv3(4, "iade", 2)
    model.to(device)

    print(f"load from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    sd = checkpoint["model"]
    ret = model.load_state_dict(sd, strict=False)
    print(f"load ret: {ret}")

    return model

def onnx_load_model(onnx_path):
    return onnxruntime.InferenceSession(onnx_path


if __name__ == "__main__":
    # model = load_model("vgg16bn_iade_4_deeplabv3_PCD.pth")
    model = 
    mask = onnx_predict(model, "img1.png", "img2.png")

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
