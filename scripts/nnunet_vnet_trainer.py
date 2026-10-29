"""An nnU-Net trainer that trains MONAI's VNet instead of nnU-Net's own U-Net.

nnU-Net keeps doing everything around the network: preprocessing, patch sampling,
augmentation, the Dice + cross-entropy loss (which skips "ignore" pixels), sliding-window
prediction and scoring. Only the network changes, so a VNet result can be compared with the
nnU-Net U-Net result on the same frames. The same trainer serves both datasets: the 2d
configuration of Dataset501 gives VNet2D, the 3d_fullres configuration of Dataset502 gives
VNet3D.

nnU-Net looks for trainers only inside its own package, both when training and when
predicting, so this file has to be linked into it once (again after the environment is
reinstalled):

    ln -s "$PWD/scripts/nnunet_vnet_trainer.py" \
        .venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/nnunet_vnet_trainer.py

    uv run --locked nnUNetv2_train 501 2d 0 -tr nnUNetTrainerVNet
    uv run --locked nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerVNet

nnUNetTrainerVNet_500epochs trains for 500 epochs instead of 1000; compare it with
nnU-Net's own nnUNetTrainer_500epochs, not with epoch 500 of a 1000-epoch run.

nnUNetTrainerVNetAniso_500epochs is a VNet3D that keeps every frame through its first two
levels. It goes with the 8-frame configurations that scripts/add_vnet_plans.py writes:

    uv run --locked nnUNetv2_train 502 3d_t8_768x512 0 -tr nnUNetTrainerVNetAniso_500epochs -p nnUNetPlans_vnet
"""

import torch
from monai.networks.nets import VNet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)


# The VNet blocks whose 5x5x5 convolutions work on levels 0 and 1, the two finest levels.
# Level 0 has the patch's full size; each level below halves it.
SHALLOW_BLOCKS = ["in_tr", "down_tr32", "up_tr64", "up_tr32", "out_tr"]

# The down- and up-sampling convolutions between levels 0, 1 and 2.
SHALLOW_DOWN_BLOCKS = ["down_tr32", "down_tr64"]
SHALLOW_UP_BLOCKS = ["up_tr64", "up_tr32"]


def use_2d_dropout(network):
    """Replace every Dropout3d layer of a VNet2D with a Dropout2d of the same rate.

    During training VNet drops whole feature maps, 50 % of them by default, in its two
    deepest down blocks and on every skip connection. MONAI 1.6.1 builds these layers as
    Dropout3d whatever spatial_dims is: VNet accepts a dropout_dim argument but never
    passes it on to its blocks. A 2D feature batch has shape (batch, channels, height,
    width). Dropout3d reads it as one unbatched (channels, depth, height, width) volume,
    so it mistakes the batch axis for the channel axis and blanks every channel of about
    half the images at once. Dropout2d blanks single channels of single images, which is
    what the dropout is meant to do.
    """
    for module in network.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Dropout3d):
                setattr(module, name, torch.nn.Dropout2d(child.p))


def keep_frames_in_shallow_levels(network):
    """Make a VNet3D treat time differently from the image plane.

    MONAI's VNet3D uses 5x5x5 convolutions everywhere and halves the frames together with
    height and width at each of its 4 downsamplings. Its very first convolution already
    mixes each frame with 2 frames on either side, which blurs the chamber walls, because
    they move from frame to frame.

    This function replaces layers of a built VNet3D in place:
    - Levels 0 and 1, the two finest levels, get 1x5x5 convolutions, so each frame is
      processed on its own. Their downsampling and upsampling convolutions get kernel and
      stride (1, 2, 2), so they halve only height and width.
    - Levels 2 to 4 get 3x5x5 convolutions, which mix each frame with one frame on either
      side. Their downsamplings still halve all three axes.

    For an 8 x 768 x 512 patch the levels are 8 x 768 x 512, 8 x 384 x 256, 8 x 192 x 128,
    4 x 96 x 64 and 2 x 48 x 32. The number of frames has to be a multiple of 4 and height
    and width multiples of 16. nnU-Net's own 3d_fullres U-Net for Dataset502 is planned
    the same way: its first two stages work within the image plane only.

    The new layers keep their channels and bias. Their padding keeps every size unchanged,
    as in MONAI.
    """
    for block_name, block in network.named_children():
        if block_name in SHALLOW_BLOCKS:
            kernel_size = (1, 5, 5)
            padding = (0, 2, 2)
        else:
            kernel_size = (3, 5, 5)
            padding = (1, 2, 2)

        for module in block.modules():
            for name, child in list(module.named_children()):
                if isinstance(child, torch.nn.Conv3d) and child.kernel_size == (5, 5, 5):
                    new_conv = torch.nn.Conv3d(
                        child.in_channels,
                        child.out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=child.bias is not None,
                    )
                    setattr(module, name, new_conv)

    for block_name in SHALLOW_DOWN_BLOCKS:
        block = getattr(network, block_name)
        old_conv = block.down_conv
        block.down_conv = torch.nn.Conv3d(
            old_conv.in_channels,
            old_conv.out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            bias=old_conv.bias is not None,
        )

    for block_name in SHALLOW_UP_BLOCKS:
        block = getattr(network, block_name)
        old_conv = block.up_conv
        block.up_conv = torch.nn.ConvTranspose3d(
            old_conv.in_channels,
            old_conv.out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            bias=old_conv.bias is not None,
        )


class nnUNetTrainerVNet(nnUNetTrainerNoDeepSupervision):
    """nnU-Net training with VNet (Milletari et al., 2016) as the network.

    The parent class turns off deep supervision: nnU-Net's U-Net also outputs smaller
    segmentations from its lower stages and adds a loss for each, while VNet has a single
    full-size output.
    """

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        """Return a VNet with as many dimensions as the planned patch.

        The patch is [768, 512] for Dataset501 (2d) and [frames, height, width] such as
        [32, 256, 192] for Dataset502 (3d_fullres). VNet halves the input 4 times, so every
        patch side has to be a multiple of 16; nnU-Net's patches are.

        num_output_channels is 5: background and the four chambers. The ignore label of
        Dataset502 gets no output of its own.

        VNet2D gets 2D dropout layers; see use_2d_dropout.
        """
        spatial_dims = len(configuration_manager.patch_size)
        network = VNet(spatial_dims=spatial_dims, in_channels=num_input_channels, out_channels=num_output_channels)

        if spatial_dims == 2:
            use_2d_dropout(network)

        return network

    def set_deep_supervision_enabled(self, enabled):
        """Do nothing: VNet has no deep supervision outputs to switch.

        nnU-Net's version sets network.decoder.deep_supervision, which only its own U-Net
        has; VNet has no decoder attribute.
        """


class nnUNetTrainerVNet_500epochs(nnUNetTrainerVNet):
    """nnUNetTrainerVNet with 500 epochs instead of 1000.

    nnU-Net lowers the learning rate from 0.01 to 0 over num_epochs, so a 500-epoch run
    ends fully annealed, while epoch 500 of a 1000-epoch run still trains at about 0.0054.
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainerVNetAniso_500epochs(nnUNetTrainerVNet_500epochs):
    """VNet3D that keeps every frame through its first two levels, for 500 epochs.

    The network is MONAI's VNet3D with the layers that keep_frames_in_shallow_levels
    replaces; channels, residual connections, dropout and BatchNorm stay as in MONAI.
    Everything else is the same as nnUNetTrainerVNet_500epochs. Use it with the 3d_t8_*
    configurations of nnUNetPlans_vnet, which scripts/add_vnet_plans.py writes.
    """

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        """Return a VNet3D whose first two levels work within the image plane."""
        network = VNet(spatial_dims=3, in_channels=num_input_channels, out_channels=num_output_channels)
        keep_frames_in_shallow_levels(network)

        return network
