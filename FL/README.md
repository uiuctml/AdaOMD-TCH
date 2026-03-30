## Federated learning experiments

### Folder structure

The scripts for data allocation, models, and methods are under `FL/scripts/`. Note that some method names used in the codes are slightly different from those in the manuscript. Here is a mapping for reference:

| manuscript | code |
| - | - |
| LS(FedAvg) | FedAvg |
| TCH | Tche |
| STCH(TERM) | STche |
| OMDgd-TCH(AFL) | AFL |
| AdaOMDgd-TCH | AFL_new |
| OMDeg-TCH | AFLeg |
| AdaOMDeg-TCH | AFLeg_new |

`FL/experiments/workflow/` contains the top-level python script, configuration files, and shell scripts for different settings. `FL/experiments/analysis/` contains the jupyter notebooks for drawing figures and tables.

### Running experiments

You can directly use the shell scripts under `FL/experiments/workflow/`, which cover all settings, methods, and variants. Here, we provide an example of running AdaOMDgd-TCH in the CIFAR-rotation setting.


1. Go to the workflow folder:
   ```bash
   cd FL/experiments/workflow
   ```
2. Append your path of the scripts folder to `$PYTHONPATH`:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:<your_path>/AdaOMD-TCH/FL/scripts"
   ```
3. Run AdaOMDgd-TCH:
   ```bash
   python -u image_main.py \
                --dataset "CIFAR10" \
                --data-mode "full" \
                --allocation "rotation" \
                --method "AFL_new" \
                --data-seed 0 \
                --train-seed 0 \
                --learning-rate-lambda 0.03 \
                --output-dir ../output/10seeds/CIFAR10_rotation/seed0/AFL_new_llr0.03
   ```
    Results will be saved in `FL/experiments/output/`.