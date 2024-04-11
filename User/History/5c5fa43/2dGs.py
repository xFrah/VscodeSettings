from collections import OrderedDict
import time
import PIL
import numpy as np
import torch
from torchvision import transforms as T
import cv2
from change_detector_model.DeepLabV3 import vgg16bn_mtf_msf_deeplabv3


def predict(model, image1_input, image2_input, device="cpu", input_size=256):
    if isinstance(image1_input, str) and isinstance(image2_input, str):
        # Load image from disk if inputs are paths
        image1 = PIL.Image.open(image1_input).convert("RGB")
        image2 = PIL.Image.open(image2_input).convert("RGB")
    elif isinstance(image1_input, np.ndarray) and isinstance(image2_input, np.ndarray):
        # Convert numpy arrays to PIL Images
        image1 = PIL.Image.fromarray(image1_input.astype("uint8"), "RGB")
        image2 = PIL.Image.fromarray(image2_input.astype("uint8"), "RGB")
    else:
        raise ValueError("Inputs must be either paths or numpy arrays")
    
    image2 = image2.resize((image2.size[0] * 2, image2.size[1]), PIL.Image.BILINEAR)
    image1 = image1.resize((image1.size[0] * 2, image1.size[1]), PIL.Image.BILINEAR)

    original_size = image1.size[::-1]
    print(f"Original size: {original_size}")

    # resize second image to the same size as the first one
    # image2 = image2.resize(image1.size, PIL.Image.BILINEAR)
    # resize both to have twice the width

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    augs = []
    augs.append(T.Resize(input_size))
    augs.append(T.ToTensor())
    augs.append(T.Normalize(mean=mean, std=std))
    transform = T.Compose(augs)

    # Convert and preprocess images
    image1 = transform(image1)
    image1 = image1.unsqueeze(0)
    image1 = image1.to(device)  # Add batch dimension and send to device
    image2 = transform(image2).unsqueeze(0).to(device)  # Add batch dimension and send to device

    model.eval()
    with torch.no_grad():
        start = time.time()
        image1, image2 = image1.to(device), image2.to(device)
        # concatenate in the channel dimension
        image = torch.cat((image1, image2), dim=1)
        output = model(image)
        if isinstance(output, OrderedDict):
            output = output["out"]
        mask_pred = torch.topk(output.data, 1, dim=1)[1][:, 0]
        mask_pred = torch.nn.functional.interpolate(mask_pred.unsqueeze(0).float(), size=original_size, mode="nearest").byte().squeeze(0)
        print(f"Inference time: {time.time() - start:.2f}s")
        # save the predicted mask
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


if __name__ == "__main__":
    model = load_model("vgg16bn_iade_4_deeplabv3_PCD.pth")
    mask = predict(model, "bg.png", "cropped.png")

    image = PIL.Image.open("bg.png")
    image = image.convert("RGB")

    # mask = np.array(mask)
    # mask = np.stack([mask, mask, mask], axis=-1)
    # mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # # apply
    # image = np.array(image)
    # masked_image = np.where(mask > 0, mask, image)
    # masked_image = PIL.Image.fromarray(masked_image)

    # 2x2 grid, img1 top left, img2 top right, mask bottom left, masked_image bottom right, use opencv
    img1 = cv2.imread("bg.png")
    img2 = cv2.imread("cropped.png")

    # resize both to have twice the width
    img2 = cv2.resize(img2, (img1.shape[1] * 2, img1.shape[0]))
    img1 = cv2.resize(img1, (img1.shape[1] * 2, img1.shape[0]))
    masked_image2 = cv2.bitwise_and(img2, img2, mask=mask)
    masked_image1 = cv2.bitwise_and(img1, img1, mask=mask)

    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    mask = cv2.resize(mask, (256, 256))
    masked_image2 = cv2.resize(masked_image2, (256, 256))
    masked_image1 = cv2.resize(masked_image1, (256, 256))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    masked_image2 = cv2.cvtColor(masked_image2, cv2.COLOR_BGR2RGB)
    masked_image1 = cv2.cvtColor(masked_image1, cv2.COLOR_BGR2RGB)

    top = np.hstack([img1, img2])
    bottom = np.hstack([mask, masked_image2])
    final = np.vstack([top, bottom])

    # show as pil image
    final = PIL.Image.fromarray(final)
    final.show()
