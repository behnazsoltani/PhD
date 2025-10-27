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



mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test1-16C-sdps-cifar100-cnn-test --comm_style pd-sgd-selection --momentum 0.9 --i1 4 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 2000 --selection bnz --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 2828 --dataset cifar100 --alpha 0.2 --KD 1 --outputFolder Output
#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph clique-ring --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test2-16C-local-cifar100-cnn --comm_style pd-sgd-selection --momentum 0.9 --i1 0 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 4000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 3789 --dataset cifar100  --outputFolder Output
#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph clique-ring --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test3-16C-local-cifar100-cnn --comm_style pd-sgd-selection --momentum 0.9 --i1 0 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 4000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 99 --dataset cifar100  --outputFolder Output
#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph clique-ring --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test4-16C-local-cifar100-cnn --comm_style pd-sgd-selection --momentum 0.9 --i1 0 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 4000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 122 --dataset cifar100  --outputFolder Output
#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph clique-ring --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test5-16C-local-cifar100-cnn --comm_style pd-sgd-selection --momentum 0.9 --i1 0 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 4000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 37 --dataset cifar100  --outputFolder Output

#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test1-16C-dsgd-cifar10-cnn --comm_style pd-sgd-selection --momentum 0.9 --i1 0 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 2000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 2828 --dataset cifar10  --outputFolder Output
#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test2-16C-dsgd-cifar10-cnn --comm_style pd-sgd-selection --momentum 0.9 --i1 0 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 2000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 3789 --dataset cifar10  --outputFolder Output
#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test3-16C-dsgd-cifar10-cnn --comm_style pd-sgd-selection --momentum 0.9 --i1 0 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 2000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 99 --dataset cifar10  --outputFolder Output
#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test4-16C-dsgd-cifar10-cnn --comm_style pd-sgd-selection --momentum 0.9 --i1 0 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 2000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 122 --dataset cifar10  --outputFolder Output
#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test5-16C-dsgd-cifar10-cnn --comm_style pd-sgd-selection --momentum 0.9 --i1 0 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 2000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 37 --dataset cifar10  --outputFolder Output