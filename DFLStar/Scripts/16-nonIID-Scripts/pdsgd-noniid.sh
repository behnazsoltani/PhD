#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=PDSGD     # sets the job name if not set from environment
#SBATCH --time=05:30:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger   # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger  # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:3
#SBATCH --ntasks=10
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi 
module load cuda/11.1.1

mpirun -np 16 --oversubscribe python Train.py  --graph grid --rows 4 --columns 4 --num_clusters 3 --name pdsgd-noniid-test1-16W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --lr 0.1 --degree_noniid 1 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description PDSGD --randomSeed 2828 --dataset cifar10  --outputFolder Output 
#mpirun -np 30 --oversubscribe python Train.py  --graph grid --rows 6 --columns 5 --num_clusters 3 --name pdsgd-noniid-test2-16W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --lr 0.1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 300  --description PDSGD --randomSeed 3789 --dataset cifar10  --outputFolder Output
#mpirun -np 30 --oversubscribe python Train.py  --graph grid --rows 6 --columns 5 --num_clusters 3 --name pdsgd-noniid-test3-16W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --lr 0.1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description PDSGD --randomSeed 99 --dataset cifar10  --outputFolder Output
#mpirun -np 30 --oversubscribe python Train.py  --graph grid --rows 6 --columns 5 --num_clusters 3 --name pdsgd-noniid-test4-16W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --lr 0.1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description PDSGD --randomSeed 122 --dataset cifar10  --outputFolder Output
#mpirun -np 30 --oversubscribe python Train.py  --graph grid --rows 6 --columns 5 --num_clusters 3 --name pdsgd-noniid-test5-16W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --lr 0.1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description PDSGD --randomSeed 37 --dataset cifar10  --outputFolder Output
 
#mpirun -np 10 python Train.py  --graph clique-ring --num_clusters 3 --name pdsgd-noniid-test2-10W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --lr 0.8 --degree_noniid 0.5 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description PDSGD --randomSeed 3789 --datasetRoot ./data --outputFolder Output
#mpirun -np 10 python Train.py  --graph clique-ring --num_clusters 3 --name pdsgd-noniid-test3-10W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --lr 0.8 --degree_noniid 0.5 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description PDSGD --randomSeed 99 --datasetRoot ./data --outputFolder Output
#mpirun -np 10 python Train.py  --graph clique-ring --num_clusters 3 --name pdsgd-noniid-test4-10W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --lr 0.8 --degree_noniid 0.5 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description PDSGD --randomSeed 122 --datasetRoot ./data --outputFolder Output
#mpirun -np 10 python Train.py  --graph clique-ring --num_clusters 3 --name pdsgd-noniid-test5-10W --comm_style pd-sgd --momentum 0.9 --i1 1 --i2 1 --lr 0.8 --degree_noniid 0.5 --noniid 1 --resSize 18 --bs 32 --epoch 300 --description PDSGD --randomSeed 37 --datasetRoot ./data --outputFolder Output
