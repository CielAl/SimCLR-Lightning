"""
Pretrain the model with CIFAR10 (or specified in dataset_name) and weaker transformation (moderate color jitter and
no blur). Strength of transformation can be specified in arguments.
"""
import argparse
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import sys
import torchvision
from simclr_lightning.models.contrast_learning.transforms import default_transform_factory
from simclr_lightning.models.lightning_data_modules.data_modules import ExampleDataModule
from simclr_lightning.models.lightning_modules.simclr_wrapper import SimCLRLightning
from simclr_lightning.models.contrast_learning.resnet_basemodel import ResNetBaseModel
from simclr_lightning.reproduce import fix_seed

print(os.getcwd())
argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Self-supervised Pretraining')
parser.add_argument('--export_folder', default='./example/output/simclr',
                    help='Export location for logs and model checkpoints')

parser.add_argument('--data_root', default='./example/data_root',
                    help='Location for the downloaded example datasets')

parser.add_argument('--dataset_name', default='cifar10',
                    help='Name of example dataset - see `torchvision.datasets` for details'
                         'For now supports cifar10 and MNIST.'
                         'Feel free to expand in `simclr_lightning.dataset.example`')

# ------------ simclr arch
parser.add_argument('--out_dim', default=128, type=int,
                    help='projection head dim (default: 128)')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
model_names = sorted(torchvision.models.resnet.__all__) 
parser.add_argument('--arch', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')

# ------------ Augmentation
parser.add_argument('--n_views', default=2, type=int,
                    help='Number of views for the augmentation.')

parser.add_argument('--patch_size', default=32, type=int,
                    help='Output size of RandomResizedCrop (default: 32 for CIFAR10)')
# jitter
parser.add_argument('--brightness', default=0.4, type=float,
                    help='brightness of ColorJitter')
parser.add_argument('--contrast', default=0.4, type=float,
                    help='contrast of ColorJitter')
parser.add_argument('--saturation', default=0.4, type=float,
                    help='saturation of ColorJitter')
parser.add_argument('--hue', default=0.1, type=float,
                    help='hue of ColorJitter')
parser.add_argument('--blur_prob', default=0, type=float,
                    help='Probability to invoke gaussian blur. 0 for cifar10. ')
# ------------ dataloader
parser.add_argument('--num_workers', default=8, type=int,
                    help='number of cpus used for DataLoader')

# ------------ trainer
parser.add_argument('--num_epochs', default=300, type=int,
                    help='max epoch')
parser.add_argument('--batch_size',
                    default=128, type=int,
                    help='batch size. '
                    'Effective batch size = N-gpu * batch size'
                    'For larger effective batch sizes, e.g., 512 or more, LARs optimizer is recommended.'
                    'In the example here only Adams is used')

parser.add_argument('--lr', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--max_t', default=90, type=int,
                    help='max_t for restarting of cosine CosineAnnealing lr_scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')
parser.add_argument('--seed', default=31415926, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--log_every_n_steps', default=25, type=int,
                    help='Log every n steps')
parser.add_argument('--gpu_index', default=[0, 1], nargs='+', type=int, help='Gpu index.')
parser.add_argument('--precision', default="16-mixed", type=str,
                    help='Precision configuration. Determine the precision of floating point used in training and '
                         'whether mixed precision is used.'
                         ' See https://lightning.ai/docs/pytorch/stable/common/trainer.html')


opt, _ = parser.parse_known_args(argv)

seed = opt.seed
fix_seed(seed, True, cudnn_deterministic=True)


if __name__ == "__main__":
    # file path curation
    export_folder = opt.export_folder
    data_root = opt.data_root
    os.makedirs(export_folder, exist_ok=True)
    os.makedirs(data_root, exist_ok=True)

    # prepare data. See `LightningDataModule`
    data_module = ExampleDataModule(root_dir=data_root, dataset_name=opt.dataset_name, batch_size=opt.batch_size,
                                    num_workers=opt.num_workers,
                                    persistent=True,
                                    seed=opt.seed)
    data_module.setup("")
    # augmentation
    aug_transforms = default_transform_factory(target_size=opt.patch_size,
                                               brightness=opt.brightness,
                                               contrast=opt.contrast,
                                               saturation=opt.saturation,
                                               hue=opt.hue,
                                               blur_prob=opt.blur_prob)

    base_model = ResNetBaseModel.build(aug_transforms, n_views=opt.n_views, model_name=opt.arch,
                                       out_dim=opt.out_dim, projection_bn=True)

    # wrap into LightningModule
    lightning_model = SimCLRLightning(base_model, lr=opt.lr, batch_size=opt.batch_size, temperature=opt.temperature,
                                      weight_decay=opt.weight_decay, prog_bar=True, max_t=opt.max_t,
                                     )

    # instantiate a csv logger, which logs the epoch or step-wise performance. In the example the logger is updated
    # each epoch
    csv_logger = CSVLogger(save_dir=export_folder, )
    # in the example the loss of validation phase is logged into the "validation_loss" entry in the LightningModule
    # Instantiate a built-in callbacl `ModelCheckpoint` to save the top-3 models with the lowest loss in all epochs.
    checkpoint_callbacks = ModelCheckpoint(monitor='validate_loss/dataloader_idx_0', save_last=True, save_top_k=3)

    trainer = L.Trainer(accelerator='gpu', devices=opt.gpu_index,
                        callbacks=[checkpoint_callbacks],
                        num_sanity_val_steps=0, max_epochs=opt.num_epochs, enable_progress_bar=True,
                        default_root_dir=export_folder, logger=csv_logger, precision=opt.precision,
                        # need to override and implement your own distributed sampler if a custom one is needed.
                        use_distributed_sampler=True, sync_batchnorm=len(opt.gpu_index) > 0,
                        log_every_n_steps=opt.log_every_n_steps)
    csv_logger.log_hyperparams(opt)
    trainer.fit(lightning_model, datamodule=data_module)
    print("Done")
