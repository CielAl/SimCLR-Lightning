"""Use a subset (e.g., 30%) of training data to finetune the model.

By default, the weights are not frozen in this tutorial.

"""
import argparse
import torch
import torchvision
from simclr_lightning.models.lightning_data_modules.data_modules import ExampleDataModule
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import sys
from simclr_lightning.reproduce import fix_seed
from simclr_lightning.models.contrast_learning.transforms import default_transform_factory, AugmentationView
from simclr_lightning.models.contrast_learning.resnet_basemodel import ResNetBaseModel
from simclr_lightning.models.lightning_modules.simclr_wrapper import SimCLRLightning
from simclr_lightning.models.lightning_modules.finetune_wrapper import FinetuneLightning
from torch import nn


argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Fine Tune')
parser.add_argument('--simclr_best_model',
                    default='./example/output/simclr/lightning_logs/version_5/checkpoints/epoch=152-step=29835.ckpt',
                    help='path to best model')
parser.add_argument('--data_root', default='./example/data_root',
                    help='Location for the downloaded example datasets')
parser.add_argument('--export_folder', default='./example/output/finetune/',
                    help='Export location for logs and model checkpoints')
parser.add_argument('--dataset_name', default='cifar10',
                    help='Name of example dataset - see `torchvision.datasets` for details'
                         'For now supports cifar10 and MNIST.'
                         'Feel free to expand in `simclr_lightning.dataset.example`')

parser.add_argument('--train_sample_ratio', default=0.3, type=float,
                    help='A fractional number between 0.0 to 1.0 indicating how much training data are sampled.')

model_names = sorted(torchvision.models.resnet.__all__)
parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--out_dim', default=128, type=int,
                    help='projection head dim (default: 128)')

# ------------ Augmentation for finetune ----------
parser.add_argument('--patch_size', default=32, type=int,
                    help='Output size of RandomResizedCrop')
# jitter
parser.add_argument('--brightness', default=0.4, type=float,
                    help='brightness of ColorJitter ')
parser.add_argument('--contrast', default=0.4, type=float,
                    help='contrast of ColorJitter ')
parser.add_argument('--saturation', default=0.4, type=float,
                    help='saturation of ColorJitter')
parser.add_argument('--hue', default=0.1, type=float,
                    help='hue of ColorJitter')

# ------------ dataloader
parser.add_argument('--num_workers', default=8, type=int,
                    help='number of cpus used for DataLoader')

# ------------ trainer
parser.add_argument('--num_epochs', default=500, type=int,
                    help='max epoch')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--lr', default=2e-4, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--max_t', default=90, type=int,
                    help='max_t for restarting of cosine CosineAnnealing lr_scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')
parser.add_argument('--seed', default=31415926, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--log_every_n_steps', default=16, type=int,
                    help='Log every n steps')
parser.add_argument('--gpu_index', default=[0, 1], nargs='+', type=int, help='Gpu index.')
parser.add_argument('--precision', default="16-mixed", type=str,
                    help='Precision configuration. Determine the precision of floating point used in training and '
                         'whether mixed precision is used.')

opt, _ = parser.parse_known_args(argv)

seed = opt.seed
fix_seed(seed, True, cudnn_deterministic=False)
generator = torch.Generator().manual_seed(seed)


if __name__ == "__main__":
    # file path curation
    best_model_path = opt.simclr_best_model
    export_folder = opt.export_folder
    os.makedirs(export_folder, exist_ok=True)

    data_root = opt.data_root
    os.makedirs(data_root, exist_ok=True)

    data_module = ExampleDataModule(root_dir=data_root, dataset_name=opt.dataset_name, batch_size=opt.batch_size,
                                    num_workers=opt.num_workers,
                                    persistent=True,
                                    seed=opt.seed,
                                    train_sample_ratio=opt.train_sample_ratio)

    # reconstruct the simclr wrapper to load its state dict
    # herein we assume the checkpoint under opt.simclr_best_model is the checkpoint of the entire `SimCLRLightning`
    # alternatively if the checkpoint is only the base model (SimCLRLightning.base_model), you may load the state dict
    # to the base model directly
    # instantiate a base model without transformation.
    # note: herein we assume in the self-supervised learning phase the transformation/augmentation operations are
    # stateless (e.g., nothing in transformation is updated in self-supervised learning loops)
    simclr_base = ResNetBaseModel.build(transforms=nn.Identity(), n_views=1, model_name=opt.arch,
                                        out_dim=opt.out_dim, projection_bn=True, return_embedding=False)
    clr_model = SimCLRLightning(simclr_base)
    checkpoint = torch.load(best_model_path, map_location='cpu')
    # we don't need the state of other things such as scheduler and optimizer, but only weights.
    clr_model.load_state_dict(checkpoint['state_dict'], strict=False)

    # custom augmentation for training in fine tune
    # enable blur and rotation
    augmentation_train = default_transform_factory(target_size=opt.patch_size,
                                                   brightness=opt.brightness,
                                                   contrast=opt.contrast,
                                                   saturation=opt.saturation,
                                                   hue=opt.hue,
                                                   blur_prob=0.5, rotation_flag=True)
    # no transformation applied in validation
    aug_val = nn.Identity()

    trans_dict = nn.ModuleDict({
        'fit': AugmentationView(augmentation_train, n_views=1),
        'validate': AugmentationView(aug_val, n_views=1),
        'test': AugmentationView(aug_val, n_views=1),
    })

    # manually call the setup here because we want to access the number of classes counted in the setup phase.
    # alternatively you may manually define it in args and Trainer will call the setup automatically.

    data_module.setup("")

    # set freeze_weight to True to only tune the classification layers.
    fine_tune_model = FinetuneLightning.build_from_lightning(
        transforms_dict=trans_dict,
        simclr_lightning=clr_model,
        num_classes=data_module.num_classes,
        freeze_weight=False,
    )
    # remove final

    csv_logger = CSVLogger(save_dir=export_folder, )
    checkpoint_callbacks = ModelCheckpoint(monitor='validate_loss', save_last=True, save_top_k=1)
    trainer = L.Trainer(accelerator='gpu', devices=opt.gpu_index,
                        callbacks=[checkpoint_callbacks],
                        num_sanity_val_steps=0, max_epochs=opt.num_epochs, enable_progress_bar=True,
                        default_root_dir=export_folder, logger=csv_logger, precision="16-mixed",
                        use_distributed_sampler=True, sync_batchnorm=len(opt.gpu_index) > 0,
                        log_every_n_steps=opt.log_every_n_steps)

    csv_logger.log_hyperparams(opt)
    trainer.fit(fine_tune_model, datamodule=data_module)
    print("Done")
