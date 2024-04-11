import datetime
import os
import time

import torch
import PIL
import torch.utils.data

from collections import OrderedDict
import numpy as np

import utils

from dataset import dataset_dict
from model import model_dict
from torchvision import transforms as T


def predict(model, image1_path, image2_path, device, input_size):
    # load image from disk
    image1 = PIL.Image.open(image1_path)
    image2 = PIL.Image.open(image2_path)
    # open as rgb
    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")

    # resize second image to the same size as the first one
    image2 = image2.resize(image1.size, PIL.Image.BILINEAR)

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
        print(f"Inference time: {time.time() - start:.2f}s")
        # save the predicted mask
        mask_pred = mask_pred.cpu().numpy().squeeze().astype(np.uint8) * 255
        mask_pred = PIL.Image.fromarray(mask_pred)
        mask_pred.show()
        return mask_pred


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = torch.device(args.device)

    args.num_classes = 2

    model = model_dict[args.model](args)
    model.to(device)

    print("load from: {}".format(args.resume))
    checkpoint = torch.load(args.resume, map_location="cpu")
    sd = checkpoint["model"]
    ret = model.load_state_dict(sd, strict=not args.test_only)
    print("load ret: {}".format(ret))

    if args.save_imgs:
        save_imgs_dir = os.path.join(args.output_dir, "img")
        os.makedirs(save_imgs_dir, exist_ok=True)
    else:
        save_imgs_dir = None

    mask = predict(model, r"img1.png", r"img2.png", device, args.input_size)

    return

    f1score = CD_evaluate(model, data_loader_test, device=device, save_imgs_dir=save_imgs_dir)
    print(f1score)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch change detection", add_help=add_help)
    parser.add_argument("--train-dataset", default="TSUNAMI", help="dataset name")
    parser.add_argument("--test-dataset", default="TSUNAMI", help="dataset name")
    parser.add_argument("--test-dataset2", default="", help="dataset name")
    parser.add_argument("--input-size", default=256, type=int, metavar="N", help="the input-size of images")
    parser.add_argument("--randomflip", default=0.5, type=float, help="random flip input")
    parser.add_argument("--randomrotate", dest="randomrotate", action="store_true", help="random rotate input")
    parser.add_argument("--randomcrop", dest="randomcrop", action="store_true", help="random crop input")
    parser.add_argument("--data-cv", default=0, type=int, metavar="N", help="the number of cross validation")

    parser.add_argument("--model", default="vgg16bn_mtf_msf_deeplabv3", help="model")
    parser.add_argument("--mtf", default="iade", help="choose branches to use")
    parser.add_argument("--msf", default=4, type=int, help="the number of MSF layers")

    parser.add_argument("--device", default="cpu", help="device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--loss", default="bi", type=str, help="the training loss")
    parser.add_argument("--loss-weight", action="store_true", help="add weight for loss")
    parser.add_argument("--opt", default="adam", type=str, help="the optimizer")
    parser.add_argument("--lr-scheduler", default="cosine", type=str, help="the lr scheduler")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--warmup", dest="warmup", action="store_true", help="warmup the lr")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, metavar="W", help="weight decay (default: 0)", dest="weight_decay")
    parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
    parser.add_argument("--resume", default="vgg16bn_iade_4_deeplabv3_PCD.pth", help="resume from checkpoint")
    parser.add_argument("--pretrained", default="", help="pretrain checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval-every", default=1, type=int, metavar="N", help="eval the model every n epoch")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", type=bool, default=True)
    parser.add_argument("--save-imgs", dest="save_imgs", type=bool, default=True, help="save the predicted mask")

    parser.add_argument("--save-local", dest="save_local", help="save logs to local", action="store_true")
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    return parser


if __name__ == "__main__":
    # os.environ["TORCH_HOME"] = '/Pretrained'
    args = get_args_parser().parse_args()
    args.output_dir = os.path.join("output", f"{args.model}_{args.train_dataset}_{args.data_cv}", f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}")

    main(args)
