import argparse
import sys
import time
sys.path.append('.')
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lib.utils import *
from lib.dataset_online import SelfDatasetsOnline
import timm

test_files = ["/home/dell/Desktop/backup/RTX6000/face_dataset/validation_data.csv"]

def parse_args():
    parser = argparse.ArgumentParser(description='Training infos')

    parser.add_argument('--model_type', type=str, default="MobileNetv2")
    parser.add_argument('--resume_checkpoints', type=str, default="/home/dell/Documents/ATF-master/checkpoint/MOBILENET_v2_0.75_0.04.pth")
    parser.add_argument('--data_type', type=str, default="300W")
    parser.add_argument('--aux_datas', type=str, default="COFW")
    parser.add_argument('--gpus', type=str, default='0', help="Set gpus environ_devices")
    parser.add_argument('--batch_size', type=int, default=50, help="batch size on one gpu")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--show_others', default=False, action="store_true")

    parser.add_argument('--image_size', type=int, default=256)

    # unused
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--model_dir', type=str, default="")

    return parser.parse_args()


def test_models(models, test_loader, criterion, model_index):
    # Initialize the loss accumulator
    losses = AverageMeter()
    losses.reset()
    # Set the model to evaluation mode
    models.eval()

    interocular_sum_nme = 0
    inter_pupil_sum_nme = 0

    # Disable gradient computation for efficiency during testing
    with torch.no_grad():
        # Loop over the test dataset
        for i, (imgs, landmarks) in enumerate(test_loader):
            # Transfer images to the GPU
            imgs = imgs.cuda(non_blocking=True)
            # Get predictions from the model
            preds_array = models(imgs)
            preds = preds_array.cuda()
            # Transfer ground truth landmarks to the GPU
            landmarks = landmarks.cuda(non_blocking=True)
            # Compute the loss between predictions and ground truth
            loss = criterion(preds, landmarks)

            # Calculate the Normalized Mean Error (NME) for the batch
            interocular_batch_nme, inter_pupil_batch_nme = compute_nme(preds.cpu(), landmarks.cpu())
            interocular_sum_nme += np.sum(interocular_batch_nme)
            inter_pupil_sum_nme += np.sum(inter_pupil_batch_nme)

            # Update the running loss
            losses.update(loss.item(), landmarks.size(0))

    # Calculate the average NME for the entire test set
    interocular_nme = interocular_sum_nme / len(test_loader.dataset)
    inter_pupil_nme = inter_pupil_sum_nme / len(test_loader.dataset)

    return losses.avg, interocular_nme, inter_pupil_nme

def get_model():
    # Create and return a MobileNetV3 model with a specified number of output classes
    model = timm.create_model("mobilenetv3_large_075", num_classes=68 * 2)
    return model

def load_model(model, gpu_nums, args):
    # Set up the model for multi-GPU training
    model = nn.DataParallel(model, range(gpu_nums)).cuda()
    devices = torch.device("cuda:0")
    model.to(devices)

    # If a checkpoint is provided, load the pretrained weights into the model
    if os.path.isfile(args.resume_checkpoints) or os.path.islink(args.resume_checkpoints):
        pretrained_dict = torch.load(args.resume_checkpoints)
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']
        model.load_state_dict(pretrained_dict)
    return model

def main(args):
    # Initialize configurations, logger, best metric tracking, and other necessary components
    cfg, logger, best_nme, model_save_dir, last_epoch, end_epoch = Init(args)
    # Set the GPUs to be used for training
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_nums = len(args.gpus.split(','))

    # Create and load the model
    model = get_model()
    model = load_model(model, gpu_nums, args)

    # Define the image transformations
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Define the loss criterion
    criterion = nn.L1Loss(size_average=True).cuda()

    # Iterate over multiple test datasets and evaluate the model
    for test_csv in test_files:
        cfg['Test_csv'] = test_csv
        logger.info("Test {}".format(test_csv))

        # Define the test dataset and dataloader
        test_dataset = SelfDatasetsOnline(cfg, is_train=False, dataset_exp=1, transforms=transform)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=(cfg["BATCHSIZE_PERGPU"]) * gpu_nums,
            shuffle=True,
            num_workers=cfg["WORKERS"],
            pin_memory=False
        )

        # Evaluate the model on the current test set
        start_test_time = time.time()
        loss, ocular_nme, pupil_nme = test_models(model, test_loader, criterion, 0)
        logger.info("test  time :{:<6.2f}s loss :{:.8f} ocular_nme:{:.5f}% pupil_nme:{:.5f}%".format(time.time() - start_test_time, loss, ocular_nme * 100.0, pupil_nme * 100.0))


if __name__ == "__main__":
    args = parse_args()
    main(args)
