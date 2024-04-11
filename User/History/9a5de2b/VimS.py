from change_detection import load_model
import torch
from torchvision import transforms as T
import PIL


def get_cd_input(width=1024):
    image1 = PIL.Image.open("media/img1.png").convert("RGB")
    image2 = PIL.Image.open("media/img2.png").convert("RGB")
    # resize both to 1024x256
    image1 = image1.resize((width, 256))
    image2 = image2.resize((width, 256))

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


model = load_model("models/vgg16bn_iade_4_deeplabv3_PCD.pth")
model.eval()
image = get_cd_input()

# onnx_program = torch.onnx.dynamo_export(
#     model,
#     image,
#     # export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
# ).save("models/cd_332.onnx")

torch.onnx.export(
    model,
    image,
    "models/cd_332.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)