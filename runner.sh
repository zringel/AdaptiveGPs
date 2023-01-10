#!/bin/bash 

C=$1
device="cuda:1"

for seed in {1..21..4} 
do
 	echo $device 
 	echo $seed 
   	python3 Teacher_student_GDNoise_fixed_sched.py --save_every 60000 --max_epochs 15000000 --lr_max 0.00025 --device $device --num_channels $C --train_seed $seed --scaling lazy --n_train 1024 --input_dim 64 &
	python3 Teacher_student_GDNoise_fixed_sched.py --save_every 60000 --max_epochs 15000000 --lr_max 0.00025 --device $device --num_channels $C --train_seed $((seed+1)) --scaling lazy --n_train 1024 --input_dim 64 &
	python3 Teacher_student_GDNoise_fixed_sched.py --save_every 60000 --max_epochs 15000000 --lr_max 0.00025 --device $device --num_channels $C --train_seed $((seed+2)) --scaling lazy --n_train 1024 --input_dim 64 &
	python3 Teacher_student_GDNoise_fixed_sched.py --save_every 60000 --max_epochs 15000000 --lr_max 0.00025 --device $device --num_channels $C --train_seed $((seed+3)) --scaling lazy --n_train 1024 --input_dim 64
        wait	
done
