# modify to your own path
export PYTHONPATH="${PYTHONPATH}:/home/nilgeoutim/AdaOMD-TCH/FL/scripts"

dataset="CIFAR10"
seeds=(0 25 37 42 53 81 119 1010 1201 2003)
preferences=(1 2 3)

# FedAvg
for preference in ${preferences[@]}; do
    for seed in ${seeds[@]}; do
        python -u image_main.py \
            --dataset ${dataset} \
            --data-mode "full" \
            --allocation "rotation" \
            --preference ${preference} \
            --method "FedAvg" \
            --data-seed ${seed} \
            --train-seed ${seed} \
            --output-dir ../output/10seeds/${dataset}_rotation_m${preference}/seed${seed}/FedAvg
    done
done

# STche
gammas=(0.01)
for preference in ${preferences[@]}; do
    for seed in ${seeds[@]}; do
        for gamma in ${gammas[@]}; do
            python -u image_main.py \
                --dataset ${dataset} \
                --data-mode "full" \
                --allocation "rotation" \
                --preference ${preference} \
                --method "STche" \
                --data-seed ${seed} \
                --train-seed ${seed} \
                --gamma ${gamma} \
                --output-dir ../output/10seeds/${dataset}_rotation_m${preference}/seed${seed}/STche_gamma${gamma}
        done
    done
done

# Tche
for preference in ${preferences[@]}; do
    for seed in ${seeds[@]}; do
        python -u image_main.py \
            --dataset ${dataset} \
            --data-mode "full" \
            --allocation "rotation" \
            --preference ${preference} \
            --method "Tche" \
            --data-seed ${seed} \
            --train-seed ${seed} \
            --output-dir ../output/10seeds/${dataset}_rotation_m${preference}/seed${seed}/Tche
    done
done

# EPO
for preference in ${preferences[@]}; do
    for seed in ${seeds[@]}; do
        python -u image_main.py \
            --dataset ${dataset} \
            --data-mode "full" \
            --allocation "rotation" \
            --preference ${preference} \
            --method "EPO" \
            --data-seed ${seed} \
            --train-seed ${seed} \
            --output-dir ../output/10seeds/${dataset}_rotation_m${preference}/seed${seed}/EPO
    done
done

# FERERO
for preference in ${preferences[@]}; do
    for seed in ${seeds[@]}; do
        python -u image_main.py \
            --dataset ${dataset} \
            --data-mode "full" \
            --allocation "rotation" \
            --preference ${preference} \
            --method "FERERO" \
            --data-seed ${seed} \
            --train-seed ${seed} \
            --output-dir ../output/10seeds/${dataset}_rotation_m${preference}/seed${seed}/FERERO
    done
done

# AFL, ExcessMTL
methods=("AFL" "ExcessMTL")
llrs=(1.0)
for preference in ${preferences[@]}; do
    for method in ${methods[@]}; do
        for seed in ${seeds[@]}; do
            for llr in ${llrs[@]}; do
                python -u image_main.py \
                    --dataset ${dataset} \
                    --data-mode "full" \
                    --allocation "rotation" \
                    --preference ${preference} \
                    --method ${method} \
                    --data-seed ${seed} \
                    --train-seed ${seed} \
                    --learning-rate-lambda ${llr} \
                    --output-dir ../output/10seeds/${dataset}_rotation_m${preference}/seed${seed}/${method}_llr${llr}
            done
        done
    done
done

# AFL_new
methods=("AFL_new")
llrs=(0.03)
for preference in ${preferences[@]}; do
    for method in ${methods[@]}; do
        for seed in ${seeds[@]}; do
            for llr in ${llrs[@]}; do
                python -u image_main.py \
                    --dataset ${dataset} \
                    --data-mode "full" \
                    --allocation "rotation" \
                    --preference ${preference} \
                    --method ${method} \
                    --data-seed ${seed} \
                    --train-seed ${seed} \
                    --learning-rate-lambda ${llr} \
                    --output-dir ../output/10seeds/${dataset}_rotation_m${preference}/seed${seed}/${method}_llr${llr}
            done
        done
    done
done