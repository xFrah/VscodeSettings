from change_detection import load_model
from torchvision import transforms as T

def get_cd_input():
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

    original_size = image1.size[::-1]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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

model = load_model('vgg16bn_iade_4_deeplabv3_PCD.pth')

