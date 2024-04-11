import onnxruntime
import PIL
import torch

def get_cd_input():
    image1 = PIL.Image.open("img1.png").convert("RGB")
    image2 = PIL.Image.open("img2.png").convert("RGB")
    # resize both to 1024x256
    image1 = image1.resize((1024, 256))
    image2 = image2.resize((1024, 256))

    original_size = image1.size[::-1]
    print(original_size)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    augs = []
    augs.append(T.Resize(256))
    augs.append(T.ToTensor())
    augs.append(T.Normalize(mean=mean, std=std))
    transform = T.Compose(augs)

    # Convert and preprocess images
    image1 = transform(image1)
    image2 = transform(image2)
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)

    image = torch.cat((image1, image2), dim=1)
    return image

image = get_cd_input()

ort_session = onnxruntime.InferenceSession("cd.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: image.cpu().numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0].shape)