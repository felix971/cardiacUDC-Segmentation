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
"""

from monai.networks.nets import VNet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
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
        """
        spatial_dims = len(configuration_manager.patch_size)
        network = VNet(spatial_dims=spatial_dims, in_channels=num_input_channels, out_channels=num_output_channels)

        return network

    def set_deep_supervision_enabled(self, enabled):
        """Do nothing: VNet has no deep supervision outputs to switch.

        nnU-Net's version sets network.decoder.deep_supervision, which only its own U-Net
        has; VNet has no decoder attribute.
        """
