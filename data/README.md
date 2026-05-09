# `data/`

Calibration observations for the QuaidSIM-v4 environment.

## Contents

```
data/QuaidSIM-v4/
    observations_TD3_2025-09-08T22-45-15.csv      14 MB  17-col, non-recurrent
    observations_RA-TD3_2025-09-08T22-45-15.csv   19 MB  25-col, recurrent + actions
    observations_RA-SAC_2025-09-09T02-53-04.csv   20 MB  25-col, recurrent + actions
```

Each row is a flat list of float observations from a single environment step,
sampled from a real training rollout of the matching policy. Column count
must match the actor input dimension:

- 17 columns -> non-recurrent TD3 / SAC, recurrent R-TD3 / R-SAC
- 25 columns -> recurrent + previous-actions: RA-TD3 / RA-SAC

## What's not here

`*.pt` calibration TensorDatasets (gitignored). PPQ needs them for ESP-DL
quantization of recurrent models because the GRU hidden-state distributions
have to be calibrated alongside the observation distribution. They are
trivially regenerable — see `drl_quant/data_generation/README.md`:

```bash
python -m drl_quant.data_generation.generate_calibration \
    -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    -s data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.csv \
    -os 25
```

## Provenance

CSVs were produced by `drl_quant.data_generation.extract_observations` from
gzipped training-buffer logs of the QuaidSIM-v4 training runs reported in the
paper. Source logs are not redistributed here; the CSVs are deterministic
slices of those logs and are sufficient for reproducing every quantization
artefact in `models/QuaidSIM-v4/`.
