"""Fixed-canvas, independent-frame preprocessing for neighbor A/B.

Register this file as nnunetv2/preprocessing/preprocessors/neighbor_ab_preprocessor.py
in the experiment's isolated environment. Set preprocessor_name to
NeighborABPreprocessor and data_identifier to
nnUNetPlans_2d_neighbor_ab_fullcanvas_v1. There is deliberately no nonzero crop,
GT normalization mask, or use of dataset foreground intensity statistics.

The run_case_npy pipeline follows nnU-Net 2.8.1's metadata/resampling/export
contract while replacing its crop and normalization steps.
"""

import numpy as np

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape


class NeighborABPreprocessor(DefaultPreprocessor):
    def run_case_npy(self, data, seg, properties, plans_manager, configuration_manager, dataset_json):
        if data.ndim != 4 or data.shape[0] != 8:
            raise ValueError("Neighbor A/B requires eight channels in [8,1,W,H] format")
        data = data.astype(np.float32, copy=True)
        has_seg = seg is not None
        if has_seg:
            if seg.shape[0] != 1 or seg.shape[1:] != data.shape[1:]:
                raise ValueError("Target segmentation must have shape [1,1,W,H]")
            seg = np.copy(seg)
        else:
            # nnU-Net expects a segmentation-shaped return even during inference.
            # This synthetic array is never used for normalization or cropping.
            seg = np.zeros((1, *data.shape[1:]), dtype=np.int8)

        transpose = [0]
        for axis in plans_manager.transpose_forward:
            transpose.append(axis + 1)
        data = data.transpose(transpose)
        seg = seg.transpose(transpose)
        if data.shape[1] != 1 or len(configuration_manager.spacing) != 2:
            raise ValueError("Neighbor A/B requires the 2D spatial configuration")
        original_spacing = []
        for axis in plans_manager.transpose_forward:
            original_spacing.append(properties["spacing"][axis])

        # Keep the complete original canvas regardless of any channel or GT values.
        full_shape = data.shape[1:]
        properties["shape_before_cropping"] = full_shape
        properties["bbox_used_for_cropping"] = []
        for size in full_shape:
            properties["bbox_used_for_cropping"].append([0, size])
        properties["shape_after_cropping_and_before_resampling"] = full_shape
        target_spacing = [original_spacing[0]] + list(configuration_manager.spacing)
        new_shape = compute_new_shape(full_shape, original_spacing, target_spacing)

        # No reduction spans channels. Full-canvas zeros contribute to each frame's
        # own mean/std. Constant frames become zeros with finite values.
        for channel in range(data.shape[0]):
            frame = data[channel]
            mean = frame.mean()
            standard_deviation = max(float(frame.std()), 1e-8)
            frame -= mean
            frame /= standard_deviation

        # Resample one channel at a time so the target cannot depend on neighbors,
        # even if a future resampler introduces channelwise batch heuristics.
        resampled_frames = []
        for channel in range(data.shape[0]):
            frame = configuration_manager.resampling_fn_data(data[channel:channel + 1], new_shape, original_spacing, target_spacing)
            resampled_frames.append(frame)
        data = np.concatenate(resampled_frames, axis=0)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)

        if has_seg:
            label_manager = plans_manager.get_label_manager(dataset_json)
            if label_manager.has_regions:
                collect_for_this = list(label_manager.foreground_regions)
            else:
                collect_for_this = list(label_manager.foreground_labels)
            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)
            properties["class_locations"] = self._sample_foreground_locations(seg, collect_for_this, verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        seg = seg.astype(np.int16 if np.max(seg) > 127 else np.int8)
        return data, seg, properties
