import datetime
import os
import time
import math

import torch
import PIL
import torch.utils.data

from collections import OrderedDict
import numpy as np

import utils

from dataset import dataset_dict
from model import model_dict
import loss
from torchvision import transforms as T


def get_scheduler_function(name, total_iters, final_lr=0):
    print("LR Scheduler: {}".format(name))
    if name == "cosine":
        return lambda step: ((1 + math.cos(step * math.pi / total_iters)) / 2) * (1 - final_lr) + final_lr
    elif name == "linear":
        return lambda step: 1 - (1 - final_lr) / total_iters * step
    elif name == "exp":
        return lambda step: (1 - step / total_iters) ** 0.9
    elif name == "none":
        return lambda step: 1
    else:
        raise ValueError(name)


def warmup(num_iter, num_warmup, optimizer):
    if num_iter < num_warmup:
        # warm up
        xi = [0, num_warmup]
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            # x['lr'] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            x["lr"] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x["lr"]])
            if "momentum" in x:
                x["momentum"] = np.interp(num_iter, xi, [0.8, 0.9])


def fix_BN_stat(module):
    classname = module.__class__.__name__
    if classname.find("BatchNorm2d") != -1:
        module.eval()
    # if classname.find('LayerNorm') != -1:
    #    module.eval()


def freeze_BN_stat(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if hasattr(model.module, "backbone"):
            print("freeze backbone BN stat")
            model.module.backbone.apply(fix_BN_stat)


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
        image1, image2 = image1.to(device), image2.to(device)
        # concatenate in the channel dimension
        image = torch.cat((image1, image2), dim=1)
        output = model(image)
        if isinstance(output, OrderedDict):
            output = output["out"]
        mask_pred = torch.topk(output.data, 1, dim=1)[1][:, 0]
        # save the predicted mask
        mask_pred = mask_pred.cpu().numpy().squeeze().astype(np.uint8) * 255
        mask_pred = PIL.Image.fromarray(mask_pred)
        mask_pred.show()
        return mask_pred


def CD_evaluate(model, data_loader, device, save_imgs_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("Prec", utils.SmoothedValue(window_size=1, fmt="{value:.3f} ({global_avg:.3f})"))
    metric_logger.add_meter("Rec", utils.SmoothedValue(window_size=1, fmt="{value:.3f} ({global_avg:.3f})"))
    metric_logger.add_meter("Acc", utils.SmoothedValue(window_size=1, fmt="{value:.3f} ({global_avg:.3f})"))
    metric_logger.add_meter("F1score", utils.SmoothedValue(window_size=1, fmt="{value:.4f} ({global_avg:.4f})"))
    header = "Test:"
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if isinstance(output, OrderedDict):
                output = output["out"]
            mask_pred = torch.topk(output.data, 1, dim=1)[1][:, 0]
            # mask_pred = (F.softmax(output.data, dim=1) > 0.5)[:, 0]
            mask_gt = (target > 0)[:, 0]
            precision, recall, accuracy, f1score = utils.CD_metric_torch(mask_pred, mask_gt)
            metric_logger.Prec.update(precision.mean(), n=len(precision))
            metric_logger.Rec.update(recall.mean(), n=len(precision))
            metric_logger.Acc.update(accuracy.mean(), n=len(precision))
            metric_logger.F1score.update(f1score.mean(), n=len(f1score))
            if save_imgs_dir:
                assert len(precision) == 1, "save imgs needs batch_size=1"
                output_pil = data_loader.dataset.get_pil(image[0], mask_gt, mask_pred)
                output_pil.save(os.path.join(save_imgs_dir, "{}_{}.png".format(utils.get_rank(), metric_logger.F1score.count)))
        metric_logger.synchronize_between_processes()

    print(
        "{} {} Total: {} Metric Prec: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
            header,
            data_loader.dataset.name,
            metric_logger.F1score.count,
            metric_logger.Prec.global_avg,
            metric_logger.Rec.global_avg,
            metric_logger.F1score.global_avg,
        )
    )
    return metric_logger.F1score.global_avg


def create_dataloader(args):
    print("arg train_dataset: {}".format(args.train_dataset))
    dataset = dataset_dict[args.train_dataset](args, train=True)
    print("train dataset: {}".format(dataset))
    dataset_test = dataset_dict[args.test_dataset](args, train=False)
    if args.test_dataset2:
        dataset_test2 = dataset_dict[args.test_dataset2](args, train=False)
    else:
        dataset_test2 = None

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        test_sampler2 = torch.utils.data.distributed.DistributedSampler(dataset_test2) if dataset_test2 else None
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler2 = torch.utils.data.SequentialSampler(dataset_test2) if dataset_test2 else None

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, collate_fn=utils.collate_fn, drop_last=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    if dataset_test2:
        data_loader_test2 = torch.utils.data.DataLoader(
            dataset_test2, batch_size=1, sampler=test_sampler2, num_workers=args.workers, collate_fn=utils.collate_fn
        )
    else:
        data_loader_test2 = None

    return dataset, train_sampler, data_loader, dataset_test, data_loader_test, dataset_test2, data_loader_test2


def main(args):
    if args.output_dir:
        print("=> creating output dir '{}'".format(args.output_dir))
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset, train_sampler, data_loader, dataset_test, data_loader_test, dataset_test2, data_loader_test2 = create_dataloader(args)

    args.num_classes = dataset.num_classes
    model = model_dict[args.model](args)
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.pretrained:
        utils.load_model(model_without_ddp, args.pretrained)

    print("load from: {}".format(args.resume))
    checkpoint = torch.load(args.resume, map_location="cpu")
    sd = checkpoint["model"]
    ret = model_without_ddp.load_state_dict(sd, strict=not args.test_only)
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
    output_dir = "output"
    args.output_dir = os.path.join(output_dir, f"{args.model}_{args.train_dataset}_{args.data_cv}", f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}")
    # save_path = "{}_{}_{}/{date:%Y-%m-%d_%H:%M:%S}".format(
    #     args.model, args.train_dataset, args.data_cv, date=datetime.datetime.now())
    # args.output_dir = os.path.join(output_dir, save_path)

    main(args)
