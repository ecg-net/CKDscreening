for class_name in A1c
do
for is_2d in True
do
for model_type in lancet lancetwide
do
for lr in 1e-2 1e-3 5e-4 1e-4
do
for conv_width in 3
do
for drop_prob in 0.0 0.1
do
for batch_norm in True
do
for first_layer_out_channels in None
do
for first_layer_kernel_size in None
do
for train_delta in 0 1
do
for train_signal_length in 4000
do
for valid_signal_length in 4000
do
for normalize_x in lead none
do
for normalize_y in False
do

command="from handle import train_wrapper; train_wrapper(class_name='${class_name}', model_type='${model_type}', is_2d=${is_2d}, lr=${lr}, binary=False, conv_width=${conv_width}, drop_prob=${drop_prob}, batch_norm=${batch_norm}, first_layer_out_channels=${first_layer_out_channels}, first_layer_kernel_size=${first_layer_kernel_size}, train_delta=${train_delta}, train_signal_length=${train_signal_length}, valid_signal_length=${valid_signal_length}, normalize_x='${normalize_x}', normalize_y=${normalize_y})"

echo
echo ${command}

echo "#!/bin/sh
. \"/home/users/jwhughes/miniconda3/etc/profile.d/conda.sh\"

# catch the SIGUSR1 signal
_resubmit() {
    echo \"\$(date): job \$SLURM_JOBID received SIGUSR1 at \$(date), re-submitting\"
    scontrol requeue \$SLURM_JOBID
}
trap _resubmit SIGUSR1

echo \"\$(date '+%d/%m/%Y %H:%M:%S')\"
echo \"${command}\"

nvidia-smi
conda activate echo

python -c \"${command}\" &

wait
" > tmp.sh

sbatch --signal=B:SIGUSR1@90 --output=../../slurm-%j.out -p owners --begin=now -N 1 -c 5 --time=0-06:00:00 --mem=40G --gres=gpu:1 tmp.sh 

done
done
done
done
done
done
done
done
done
done
done
done
done
done
