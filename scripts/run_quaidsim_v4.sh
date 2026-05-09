#!/usr/bin/env bash
#
# End-to-end pipeline for QuaidSIM-v4: TorchScript .dat -> ONNX ->
# (dynamic-quant ONNX, ESP-DL .espdl). Re-running this script regenerates
# every artefact under models/QuaidSIM-v4/{onnx,onnx-quant,esp-dl}/ from the
# bundled .dat checkpoints and observation CSVs.
#
# Run from the repo root, with the package installed (pip install -e .).

set -euo pipefail

cd "$(dirname "$0")/.."

OBS_TD3=data/QuaidSIM-v4/observations_TD3_2025-09-08T22-45-15.csv
OBS_RA_TD3=data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.csv
OBS_RA_SAC=data/QuaidSIM-v4/observations_RA-SAC_2025-09-09T02-53-04.csv

CALIB_TD3=${OBS_TD3%.csv}.pt
CALIB_RA_TD3=${OBS_RA_TD3%.csv}.pt

GRU_RA=models/rnn/rnn_Quaid_RA-64.dat

CPP=models/QuaidSIM-v4/cpp
ONNX=models/QuaidSIM-v4/onnx

# ---------- Step 2: TorchScript -> ONNX ---------------------------------------

echo "=== Step 2: ONNX export ==="

# Non-recurrent
python -m drl_quant.onnx_export.export_actor -i $CPP/act_net_QuaidSIM-v4_TD3_+225.827_750000.dat -s $OBS_TD3
python -m drl_quant.onnx_export.export_actor -i $CPP/act_net_QuaidSIM-v4_SAC_+238.823_775000.dat -s $OBS_TD3

# Recurrent (Aug-GRU) — deployment path
for ckpt in \
    act_net_QuaidSIM-v4_RA-TD3_+364.117_475000.dat \
    act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.dat \
    act_net_QuaidSIM-v4_RA-TD3_+614.464_500000.dat \
    act_net_QuaidSIM-v4_RA-SAC_+299.788_850000.dat; do
    python -m drl_quant.onnx_export.export_aug_actor \
        -i $CPP/$ckpt -s $OBS_RA_TD3 -g $GRU_RA
done

# Recurrent (native nn.GRU) — baseline only
python -m drl_quant.onnx_export.export_native_gru_actor \
    -i $CPP/act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.dat -s $OBS_RA_TD3 -g $GRU_RA
python -m drl_quant.onnx_export.export_native_gru_actor \
    -i $CPP/act_net_QuaidSIM-v4_RA-SAC_+299.788_850000.dat -s $OBS_RA_TD3 -g $GRU_RA

# ---------- Step 1 (calibration .pt) ------------------------------------------
# Re-run after step 2 because we need an ONNX model to roll out trajectories
# against. The .pt files are gitignored.

echo "=== Step 1: calibration set ==="

python -m drl_quant.data_generation.generate_calibration \
    -i $ONNX/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    -s $OBS_RA_TD3 -os 25

# Non-recurrent calibration: roll out an Aug- model just to get a TensorDataset
# whose first column matches the 17-dim observation. Step 4a only reads the
# observation column.
python -m drl_quant.data_generation.generate_calibration \
    -i $ONNX/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    -s $OBS_TD3 -os 17

# ---------- Step 3: dynamic ONNX quantization ---------------------------------

echo "=== Step 3: dynamic ONNX quantization ==="

for f in $ONNX/*.onnx; do
    python -m drl_quant.onnx_dynamic_quantize.quantize -i "$f"
done

# ---------- Step 4: ESP-DL int8 quantization ----------------------------------

echo "=== Step 4: ESP-DL quantization ==="

# Non-recurrent
python -m drl_quant.espdl_quantize.quantize_actor \
    -i $ONNX/act_net_QuaidSIM-v4_TD3_+225.827_750000.onnx -s $CALIB_TD3 -os 17
python -m drl_quant.espdl_quantize.quantize_actor \
    -i $ONNX/act_net_QuaidSIM-v4_SAC_+238.823_775000.onnx -s $CALIB_TD3 -os 17

# Recurrent (Aug-GRU only — native-GRU does not survive ESP-DL quantization)
for ckpt in \
    aug_act_net_QuaidSIM-v4_RA-TD3_+364.117_475000.onnx \
    aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    aug_act_net_QuaidSIM-v4_RA-TD3_+614.464_500000.onnx \
    aug_act_net_QuaidSIM-v4_RA-SAC_+299.788_850000.onnx; do
    python -m drl_quant.espdl_quantize.quantize_recurrent \
        -i $ONNX/$ckpt -s $CALIB_RA_TD3 -os 25
done

echo
echo "Pipeline complete. Artefacts:"
echo "  ONNX:           models/QuaidSIM-v4/onnx/"
echo "  Dynamic-quant:  models/QuaidSIM-v4/onnx-quant/"
echo "  ESP-DL:         models/QuaidSIM-v4/esp-dl/"
