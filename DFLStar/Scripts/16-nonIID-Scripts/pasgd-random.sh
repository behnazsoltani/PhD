module load openmpi 
module load cuda/11.1.1

#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test1-16C-pasgd-cifar100-cnn-motivation --comm_style pd-sgd-selection --momentum 0.9 --i1 4 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 4000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 2828 --dataset cifar100 --KD 0 --PS 0 --outputFolder Output

mpirun -np 16 --oversubscribe python Train_pull_logit_time.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test1-16C-pasgdrandom-cifar100-cnn-motivation --comm_style pd-sgd-selection --momentum 0.9 --i1 4 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 1 --resSize 18 --bs 32 --epoch 4000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 2828 --dataset cifar100 --KD 0 --PS 1 --outputFolder Output


#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test1-16C-pasgd-cifar100-cnn-IID --comm_style pd-sgd-selection --momentum 0.9 --i1 4 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 0 --resSize 18 --bs 32 --epoch 4000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 2828 --dataset cifar100 --KD 0 --PS 0 --outputFolder Output
#
#mpirun -np 16 --oversubscribe python Train_pull_logit.py --graph grid --rows 4 --columns 4 --num_clusters 4 --name pdsgd-noniid-test1-16C-pasgdrandom-cifar100-cnn-IID --comm_style pd-sgd-selection --momentum 0.9 --i1 4 --i2 1 --lr 0.01 --customLR 1 --degree_noniid 0.9 --noniid 0 --resSize 18 --bs 32 --epoch 4000 --selection random --utility_metric cosine_similarity_weight --description PDSGD --randomSeed 2828 --dataset cifar100 --KD 0 --PS 1 --outputFolder Output