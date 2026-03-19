# modify to your own path
export PYTHONPATH="${PYTHONPATH}:/home/nilgeoutim/tmlr_code/AdaOMD-TCH-master/synthetic"

problems=("VLMOP2" "F1" "F2" "F3" "F4" "F5" "F6")
seeds="0,19,42"

solver_names=("agg_ls" "agg_tche" "pmtl" "epo" "ferero")
step_size=0.01
for problem in ${problems[@]}; do
  for solver_name in ${solver_names[@]}; do
    python run_syn_discrete.py \
    --problem-name $problem \
    --solver-name $solver_name \
    --epoch 1000 \
    --step-size $step_size \
    --seeds $seeds \
    --draw-fig True
  done
done

solver_names=("agg_omdgdtche" "agg_gomdgdtche" "agg_omdegtche" "agg_gomdegtche" "excessmtl")
step_size=0.02
eta=1.0
for problem in ${problems[@]}; do
  for solver_name in ${solver_names[@]}; do
    python run_syn_discrete.py \
      --problem-name $problem \
      --solver-name $solver_name \
      --epoch 1000 \
      --step-size $step_size \
      --seeds $seeds \
      --eta $eta \
      --draw-fig True
  done
done

solver_names=("agg_softtche")
step_size=0.02
mu=0.01
for problem in ${problems[@]}; do
  for solver_name in ${solver_names[@]}; do
    python run_syn_discrete.py \
      --problem-name $problem \
      --solver-name $solver_name \
      --epoch 1000 \
      --step-size $step_size \
      --seeds $seeds \
      --mu $mu \
      --draw-fig True
  done
done

solver_names=("gm_mgda" "gm_crmogm" "gm_moco")
step_size=0.02
eta=1.0
pho=0.01
for problem in ${problems[@]}; do
  for solver_name in ${solver_names[@]}; do
    python3 run_syn_discrete.py \
      --problem-name $problem \
      --solver-name $solver_name \
      --epoch 1000 \
      --step-size $step_size \
      --seeds $seeds \
      --eta $eta \
      --pho $pho \
      --draw-fig True
  done
done