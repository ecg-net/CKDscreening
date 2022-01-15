for target in hemoglobin
do
    for lr in 1e-2 5e-3 1e-3
    do
        sbatch run_lab.sh ${target} ${lr} lancet
        echo run_lab.sh ${target} ${lr} lancet
        echo
    done
done
