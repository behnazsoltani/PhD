# Beyond Parameters: Locally-Guided Knowledge Distillation for Decentralized Federated Learning

Install the dependencies using:

```bash
pip install -r requirements.txt
```

Run the following command to start training BlendDFL with 50 clients on CIFAR-10 using a ring topology:

```bash
python dfl.py --client_num_in_total 50 --comm_round 1000 --partition_method dir --partition_alpha 0.3 --algorithm blenddfl --model cnn_cifar10 --dataset cifar10 --apply_lr_decay 
  --topology ring --lr 0.01 --round_number_evaluation 10 --temperature 3 --kd_weight 25 --gpu 0
```
# Key Arguments:

| Argument                    | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--client_num_in_total`    | Total number of clients                                                     |
| `--comm_round`             | Number of communication rounds                                              |
| `--partition_method`       | Data partitioning strategy (`dir` = Dirichlet)                              |
| `--partition_alpha`        | Dirichlet alpha for non-IID data distribution                               |
| `--algorithm`              | DFL algorithm to run (`blenddfl`, etc.)                                     |
| `--model`                  | Model architecture (`cnn_cifar10`, `cnn_cifar100`, etc.)                        |
| `--dataset`                | Dataset to use (`CIFAR-10`, `CIFAR-100`, etc.)                                   |
| `--apply_lr_decay`         | Use learning rate decay (flag; add to enable)                               |
| `--topology`               | Communication topology (`ring`, `grid`, etc.)                               |
| `--lr`                     | Learning rate                                                               |
| `--round_number_evaluation`| Frequency of evaluation (every N rounds)                                    |
| `--temperature`            | Temperature for knowledge distillation                                      |
| `--kd_weight`              | Weight of distillation loss                                                 |                                          |
| `--gpu`                    | GPU index to use                                             |

