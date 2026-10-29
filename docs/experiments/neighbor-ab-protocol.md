# Neighbor-content A/B experiment

## Question and scope

Does real adjacent-frame image content improve target-frame segmentation under a fixed VNet3D-T8 training protocol? This does not test temporal ordering or compare architecture families. Existing 2D scores are contextual references, not causal controls.

## Fixed comparison

- A: repeat target frame eight times AFTER all image preprocessing and augmentation.
- B: eight actual frames at offsets [-4,-3,-2,-1,0,1,2,3]. Clamp each index independently to the same video's boundaries. Target index is always 4.
- Identical full 3D anisotropic VNet, parameters, paired initial state, optimizer, batch size, sampling and augmentation sequence. Only the input content differs.
- A common 8-channel 2D cache represents frames. A wrapper reshapes [N,8,H,W] to [N,1,8,H,W], runs VNet3D, and returns only output frame 4. This retains 3D computation while reusing nnU-Net's 2D losses, spatial sliding-window inference and export.
- Per-frame image-only normalization. Fixed original canvas; no cross-channel nonzero bounding box, GT-based mask, or full-video intensity statistics. A's complete input must be invariant to replacement of all seven neighbors.
- Synchronized 2D spatial augmentation across the eight channels; no time reversal, temporal interpolation, or temporal test-time augmentation. Intensity operations must not derive parameters/statistics from neighboring channels.
- Only target-frame GT is exposed to loss. Five output classes: background/LV/LA/RA/RV. Preserve declared ignore semantics.
- Original baseline_v1 group split: 970 training targets and the same 218 validation targets. No test-set access for design choices. Keep all known difficult validation cases.
- Candidate spatial patch [768,512] in nnU-Net orientation, batch 2. A bounded real forward/backward resource check must confirm it fits; shared fallback [512,384] only before formal training, recorded here.
- SGD learning rate 0.01, momentum 0.99, Nesterov, weight decay 3e-5; Dice+CE; 500 epochs x 250 updates; polynomial schedule. No architecture selection or early stopping based on validation.
- Primary endpoint: final checkpoint, complete 218-frame validation, foreground class macro Dice with per-frame and per-class details; identical spatial inference, no temporal averaging.

## Parallel execution and compute limit

Each host runs a matched A/B pair, sequentially on its GPU. Ubuntu uses seed 2026, A then B. Colab uses seed 2027, B then A. Four total runs, no automatic additional search. Cross-hardware A-versus-B alone would confound the contrast; the estimand is each within-host B-A difference, reported separately and together. Two seed pairs are an initial robustness check, not a precise estimate of seed variance.

Before launch: semantic unit tests, actual-data ID/target/label checks, paired initial-state hashes, bounded GPU forward/backward and predictor smoke check, independent review. Any material defect blocks long training until fixed.

Results, logs, checkpoints, exact code/config/data fingerprints must survive disconnects. Colab runs from local /content and writes durable results to Drive, then releases its runtime only after both runs and complete validation are saved and verified. Ubuntu's runner exits normally after its pair.

## Implementation tasks

1. Data export and preprocessing: scripts/export_neighbor_ab.py, scripts/neighbor_ab_preprocessor.py; tests/test_neighbor_ab_data.py. Produce Dataset503_CardiacUDCNeighborAB with eight image channels and existing target case IDs, split manifest, unique cache ID, fixed-canvas preprocessor.
2. Network/training adapter: scripts/nnunet_neighbor_ab_trainer.py; tests/test_neighbor_ab_trainer.py. Reuse T8 architecture helper. Paired deterministic single-thread sampling/augmentation; isolated RNG/checkpoint provenance; A/B trainer names; target-only wrapper.
3. Orchestration: setup/registration scripts, run-pair runner, locked bundle, resource/semantic audits, Colab notebook, independent review, actual launch.

## Interface agreement

Data exports original frame orientation exactly as export_nnunet.py. Preprocessed images have channels in offset order, target channel 4. Default nnU-Net 2D loader supplies [N,8,H,W]; target [N,1,H,W]. The network wrapper controls A/B mode. Config is named 2d, has batch 2 and a unique data identifier. Seed is explicit in environment NEIGHBOR_AB_SEED and persisted in every run manifest/checkpoint. Output directories must distinguish seed, mode and host.

## Final implementation decisions (before launch)

- The real cache preserves the complete canvas [8,1,width,height]. Spatial training patches remain [768,512], batch2; the measured 4090 memory screen passed without fallback.
- Each training sample gets shared X/Y flip draws (probability .5 per axis), rotation (probability .2, uniform -15 to 15 degrees), scale (probability .2, uniform .85 to1.25), scalar gain (probability .15, uniform .75 to1.25), and scalar offset (probability .15, uniform -.1 to.1). Image interpolation is bilinear, labels nearest; no pixel statistics drive augmentation. Loader foreground sampling uses only targetGT, with the usual .33 foreground fraction.
- Inference is Gaussian spatial sliding-window, step .5, no mirroring. Future neighbors +1,+2,+3 make this an offline segmentation experiment, not causal realtime inference.
- Strict deterministic PyTorch algorithms, cuDNN benchmark off, deterministic cuDNN on, TF32 off, and separate checkpointable train/validation/model random streams. nnU-Net CLI overrides are reasserted at trainer entry.
- PyTorch2.14.1's CUDA 2D NLL kernel rejects strict determinism. Replace only CE with the mathematically equivalent log_softmax/gather/masked-mean formulation, retaining nnU-Net Dice. CPU loss/gradient equivalence and strict CUDA checkpoint continuation are regression-tested.
- Save every10epochs and at the end. Each complete run must record125000 training batches and25000 sampled validation batches; the pair must match cumulative sample/augmentation/target hashes and initial state/RNG hashes.
- Compare final checkpoint on all218frames. Recompute per-class and foreground Dice aggregates from per-case results, reject missing/extra files, invalid metric values, wrongmode/seed/updatecounts or changedcode/data. EveryselectedGTframecontainsallfourforegroundclasses, so per-case Dice is finite.
- Prior2D results are descriptive context only: they used different architecture/preprocessing/inference. The primary result is each within-host B-A difference and their arithmetic mean, with no precise seed-variance claim fromtwo pairs.
