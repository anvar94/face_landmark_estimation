import argparse
import sys

sys.path.append('.')
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lib.utils import *
from lib.dataset_online import SelfDatasetsOnline
from losses.wing_loss import WingLoss
import timm
import wandb



def parse_args():
    parser = argparse.ArgumentParser(description='Training infos')
    parser.add_argument('--data_type', type=str, default="300W")
    parser.add_argument('--dataset_exp', type=int, default=1)
    parser.add_argument('--model_type', type=str, default="resnet18")
    parser.add_argument('--resume_checkpoints', type=str, default="")
    parser.add_argument('--pretrained', type=str, default="")
    parser.add_argument('--lr_reduce_patience', type=int, default=0, help="use ReduceLROnPlateau")
    parser.add_argument('--model_dir', type=str, default="checkpoint",
                        help="model save in checkpoint/data_type/model_dir/*_checkpoint.pth")
    parser.add_argument('--load_epoch', type=int, default=0,
                        help="wheather load epoch and lr when restore checkpoint. 0 is no, others is yes")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpus', type=str, default='0', help="Set gpus environ_devices")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size on one gpu")
    parser.add_argument('--loss_type', type=str, default='WING')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=224)
    return parser.parse_args()


def main(args):
    # Initialize Weights and Biases (wandb) for experiment tracking and logging
    wandb.init(project="ResNet18 224 * 224 _Synthetic_Unbalanced",
               config={"architecture": args})

    # Initialize configurations, logger, best metric tracking, and other necessary components
    cfg, logger, best_nme, model_save_dir, last_epoch, end_epoch = Init(args)

    # Set GPU devices for training
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_nums = len(args.gpus.split(','))

    # Initialize the model based on provided type (ResNet18 in this case)
    model = None
    if args.model_type in ["resnet18"]:
        model = timm.create_model("resnet18", num_classes=68 * 2)

    # Initialize the loss function based on the type provided
    if args.loss_type in ["WING", "wing"]:
        criterion = WingLoss().cuda()

    # Set up the model for multi-GPU training
    model = nn.DataParallel(model, range(gpu_nums)).cuda()
    devices = torch.device("cuda:0")
    model.to(devices)

    # Define the optimizer for the model's parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-6,
    )

    # If there's a checkpoint provided, restore model from the checkpoint
    if os.path.exists(args.resume_checkpoints) or os.path.islink(args.resume_checkpoints):
        checkpoint = torch.load(args.resume_checkpoints)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Restore epoch {} from {}".format(checkpoint['epoch'], args.resume_checkpoints))
        best_nme = checkpoint['best_nme']

    # Learning rate scheduler setup. Either a MultiStepLR or ReduceLROnPlateau is used based on arguments
    if args.lr_reduce_patience:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=args.lr_reduce_patience, threshold=1e-4
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg["LR_STEP"],
            0.1, last_epoch - 1
        )

    # Image preprocessing transformations
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Define training and testing datasets and dataloaders
    train_dataset = SelfDatasetsOnline(cfg, is_train=True, transforms=transform, dataset_exp=args.dataset_exp)
    test_dataset = SelfDatasetsOnline(cfg, is_train=False, transforms=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size * gpu_nums,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=(args.batch_size // 2) * gpu_nums,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # Start the training loop
    for epoch in range(last_epoch, end_epoch):
        logger.info("Use {} train epoch {}".format(lr_repr(optimizer), epoch))

        # Train the model for one epoch
        start_time = time.time()
        loss, ocular_nme, pupil_nme = train(model, train_loader, criterion, optimizer, epoch)
        logger.info("{}'epoch train time :{:<6.2f}s loss :{:.8f} ocular_nme:{:.5f}% pupil_nme:{:.5f}%".format(epoch, time.time() - start_time, loss, ocular_nme * 100.0, pupil_nme * 100.0))

        # Validate the model on the test set
        start_test_time = time.time()
        loss, ocular_nme, pupil_nme = test(model, test_loader, criterion)
        logger.info("{}'epoch test  time :{:<6.2f}s loss :{:.8f} ocular_nme:{:.5f}% pupil_nme:{:.5f}%".format(epoch, time.time() - start_test_time, loss, ocular_nme * 100.0, pupil_nme * 100.0))

        # Log results to Weights and Biases
        wandb.log({"epoch": epoch, "val_nme": pupil_nme, "ocular_nme": ocular_nme, "val_loss": loss, "training_loss": trainig_loss})

        # Save model checkpoint if performance improved
        if ocular_nme < best_nme['nme']:
            best_nme = {'epoch': epoch, 'loss': loss, 'nme': ocular_nme, 'pupil_nme': pupil_nme}
            logger.info('epoch {} reach better, save {}_checkpoint.pth'.format(epoch, epoch))
            save_checkpoint(model.state_dict(), best_nme, optimizer.state_dict(), model_save_dir)

        # Adjust learning rate
        if args.lr_reduce_patience:
            lr_scheduler.step(loss)
        else:
            lr_scheduler.step()

    # Synchronize Weights and Biases with TensorBoard
    wandb.init(sync_tensorboard=True)
if __name__ == "__main__":
    args = parse_args()
    main(args)
