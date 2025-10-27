module load openmpi 
module load cuda/11.1.1

mpirun -np 16 --oversubscribe python Train_pull_logit_time.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test1-16C-SDPS-cifar10-cnn-lastfc- --comm_style pd-sgd-selection --momentum 0.9 --i1 4 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 2000 --selection bnz --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 2828 --dataset cifar10 --lambda_kd 0.1 --KD 0 --PS 0 --layer last_fc --outputFolder Output
