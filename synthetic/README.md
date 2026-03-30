## Synthetic experiments

This folder is built upon the [LibMOON](https://github.com/xzhang2523/libmoon) library. Note that some method names used in the codes are slightly different from those in the manuscript. Here is a mapping for reference:

| manuscript | code |
| - | - |
| LS | agg_ls |
| TCH | agg_tche |
| STCH | agg_softtche |
| OMDgd-TCH | agg_omdgdtche |
| AdaOMDgd-TCH | agg_gomdgdtche |
| OMDeg-TCH | agg_omdegtche |
| AdaOMDeg-TCH | agg_gomdegtche |
| MGDA | gm_gmda |
| CR-MOGM | gm_crmogm |
| Moco | gm_moco |

To run the experiments, use `synthetic/libmoonRAW/tester/run_syn_discrete.sh`. Here, we provide an example of running AdaOMDgd-TCH for the VLMOP2 problem:

1. Go to the tester folder:
   ```bash
   cd synthetic/libmoonRAW/tester
   ```
2. Append your path of the synthetic folder to `$PYTHONPATH`:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:<your_path>/AdaOMD-TCH/synthetic"
   ```
3. Run AdaOMDgd-TCH:
   ```bash
   python run_syn_discrete.py \
      --problem-name "VLMOP2" \
      --solver-name "agg_gomdgdtche" \
      --epoch 1000 \
      --step-size 0.02 \
      --seeds "0,19,42" \
      --eta 1.0 \
      --draw-fig True
   ```
   Results will be saved in `synthetic/libmoonRAW/Output/discrete/`.