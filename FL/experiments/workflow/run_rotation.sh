# modify to your own path
export PYTHONPATH="${PYTHONPATH}:/home/nilgeoutim/AdaOMD-TCH/fairFL/scripts"

dataset="CIFAR10"
seeds=(0 25 37 42 53 81 119 1010 1201 2003)

# FedAvg
for seed in ${seeds[@]}; do
    python -u image_main.py \
        --dataset ${dataset} \
        --data-mode "full" \
        --allocation "rotation" \
        --method "FedAvg" \
        --data-seed ${seed} \
        --train-seed ${seed} \
        --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/FedAvg
done

# STche
gammas=(0.01 0.03 0.1 0.3 1.0 3.0 10.0)
for seed in ${seeds[@]}; do
    for gamma in ${gammas[@]}; do
        python -u image_main.py \
            --dataset ${dataset} \
            --data-mode "full" \
            --allocation "rotation" \
            --method "STche" \
            --data-seed ${seed} \
            --train-seed ${seed} \
            --gamma ${gamma} \
            --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/STche_gamma${gamma}
    done
done
# STche_momentum
for seed in ${seeds[@]}; do
    for gamma in ${gammas[@]}; do
        python -u image_main.py \
            --dataset ${dataset} \
            --data-mode "full" \
            --allocation "rotation" \
            --method "STche" \
            --data-seed ${seed} \
            --train-seed ${seed} \
            --gamma ${gamma} \
            --momentum 1 \
            --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/STche_gamma${gamma}_momentum
    done
done

# Tche
for seed in ${seeds[@]}; do
    python -u image_main.py \
        --dataset ${dataset} \
        --data-mode "full" \
        --allocation "rotation" \
        --method "Tche" \
        --data-seed ${seed} \
        --train-seed ${seed} \
        --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/Tche
done

# qFFL
qs=(0.1 0.5 1.0 5.0 10.0)
for seed in ${seeds[@]}; do
    for q in ${qs[@]}; do
        python -u image_main.py \
            --dataset ${dataset} \
            --data-mode "full" \
            --allocation "rotation" \
            --method "qFFL" \
            --data-seed ${seed} \
            --train-seed ${seed} \
            --q ${q} \
            --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/qFFL_q${q}
    done
done

# PropFair
bases=(2.0 3.0 4.0 5.0)
for seed in ${seeds[@]}; do
    for base in ${bases[@]}; do
        python -u image_main.py \
            --dataset ${dataset} \
            --data-mode "full" \
            --allocation "rotation" \
            --method "PropFair" \
            --data-seed ${seed} \
            --train-seed ${seed} \
            --base ${base} \
            --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/PropFair_base${base}
    done
done

# FedMGDA
epsilons=(0.05 0.1 0.5 1.0)
for seed in ${seeds[@]}; do
    for epsilon in ${epsilons[@]}; do
        python -u image_main.py \
            --dataset ${dataset} \
            --data-mode "full" \
            --allocation "rotation" \
            --method "FedMGDA" \
            --data-seed ${seed} \
            --train-seed ${seed} \
            --epsilon ${epsilon} \
            --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/FedMGDA_epsilon${epsilon}
    done
done

# FedFV
alphas=(0.1 0.2 0.5)
fedfv_taus=(0 1 3 10)
for seed in ${seeds[@]}; do
    for alpha in ${alphas[@]}; do
        for fedfv_tau in ${fedfv_taus[@]}; do
            python -u image_main.py \
                --dataset ${dataset} \
                --data-mode "full" \
                --allocation "rotation" \
                --method "FedFV" \
                --data-seed ${seed} \
                --train-seed ${seed} \
                --alpha ${alpha} \
                --fedfv-tau ${fedfv_tau} \
                --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/FedFV_alpha${alpha}_tau${fedfv_tau}
        done
    done
done

# EPO
for seed in ${seeds[@]}; do
    python -u image_main.py \
        --dataset ${dataset} \
        --data-mode "full" \
        --allocation "rotation" \
        --method "EPO" \
        --data-seed ${seed} \
        --train-seed ${seed} \
        --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/EPO
done

# FERERO
for seed in ${seeds[@]}; do
    python -u image_main.py \
        --dataset ${dataset} \
        --data-mode "full" \
        --allocation "rotation" \
        --method "FERERO" \
        --data-seed ${seed} \
        --train-seed ${seed} \
        --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/FERERO
done

# AFL, AFLeg, AFL_new, AFLeg_new, ExcessMTL, AdaExcessMTL
methods=("AFL" "AFLeg" "AFL_new" "AFLeg_new" "ExcessMTL" "AdaExcessMTL")
llrs=(0.001 0.003 0.01 0.03 0.1 0.3 1.0)
for method in ${methods[@]}; do
    for seed in ${seeds[@]}; do
        for llr in ${llrs[@]}; do
            python -u image_main.py \
                --dataset ${dataset} \
                --data-mode "full" \
                --allocation "rotation" \
                --method ${method} \
                --data-seed ${seed} \
                --train-seed ${seed} \
                --learning-rate-lambda ${llr} \
                --output-dir ../output/10seeds/${dataset}_rotation/seed${seed}/${method}_llr${llr}
        done
    done
done