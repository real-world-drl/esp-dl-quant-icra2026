# `models/`

All model artefacts for QuaidSIM-v4, plus the shared GRU TorchScript that
gets transplanted into Aug-GRU during ONNX export.

## Layout

```
models/
    rnn/
        rnn_Quaid_RA-64.dat                3-layer, hidden 64, no bias.
                                           Shared across all RA- actors.
    QuaidSIM-v4/
        cpp/                               Inputs: TorchScript .dat actors
            act_net_*_TD3_*.dat            non-recurrent
            act_net_*_SAC_*.dat            non-recurrent
            act_net_*_RA-TD3_*.dat         recurrent + previous-actions
            act_net_*_RA-SAC_*.dat         recurrent + previous-actions
        onnx/                              Step 2 outputs
            act_net_*.onnx                 from export_actor
            aug_act_net_*.onnx             from export_aug_actor (deployment)
            with_gru_act_net_*.onnx        from export_native_gru_actor (baseline)
        onnx-quant/                        Step 3 outputs (host-side benchmark)
            *_qd.onnx
        esp-dl/                            Step 4 outputs (deployment)
            *.espdl                        flash this to the MCU
            *.native + *.cfg               PPQ NATIVE format for offline checks
            *.info / *.json                PPQ-emitted metadata
```

## Filename conventions (load-bearing)

| Prefix / marker          | Meaning                                                    |
|--------------------------|------------------------------------------------------------|
| `act_net_`               | actor head only                                            |
| `aug_act_net_`           | actor + Aug-GRU (export_aug_actor output)                  |
| `with_gru_act_net_`      | actor + native `nn.GRU` (export_native_gru_actor output)   |
| `_TD3_` / `_SAC_`        | algorithm; `TD3` substring drives head selection in step 2 |
| `R-` / `RA-`             | recurrent (R-) or recurrent + actions (RA-, 25-dim obs)    |
| `+<reward>_<step>`       | training metric trailer; left as-is for traceability       |

## Bundled checkpoints

Six TorchScript actors are shipped under `cpp/`:

- `act_net_QuaidSIM-v4_TD3_+225.827_750000.dat`
- `act_net_QuaidSIM-v4_SAC_+238.823_775000.dat`
- `act_net_QuaidSIM-v4_RA-TD3_+364.117_475000.dat`
- `act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.dat`
- `act_net_QuaidSIM-v4_RA-TD3_+614.464_500000.dat`
- `act_net_QuaidSIM-v4_RA-SAC_+299.788_850000.dat`

The corresponding step-2/3/4 artefacts are also shipped, so you can compare
freshly regenerated outputs byte-for-byte against the versions used in the
paper.
