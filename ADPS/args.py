import argparse

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
  
     
    
    # Training settings
    parser.add_argument('--model', type=str, default='resnet18',
                        help="network architecture, supporting 'cnn_cifar10', 'cnn_cifar100', 'resnet18', 'vgg11'")

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training')

    parser.add_argument('--momentum', type=float, default=0,
                        help='momentum')
    parser.add_argument('--dfedavgm_momentum', type=float, default=0.9,
                        help='momentum')

    parser.add_argument('--data_dir', type=str, default='/disks/storage01/scratch/Behnaz/BNZ-framework/data/',
                        help='data directory, please feel free to change the directory to the right place')

    parser.add_argument('--partition_method', type=str, default='dir', metavar='N',
                        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
                             "one called 'n_cls' short for how many classes allocated for each client"
                             "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution")

    parser.add_argument('--partition_alpha', type=float, default=0.3,
                        help='available parameters for data partition method')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='local batch size for training')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
                        
    parser.add_argument('--min_lr', type=float, default=1e-4, metavar='min_LR',
                        help='minimum learning rate ')

    parser.add_argument('--lr_decay', type=float, default=0.998, metavar='LR_decay',
                        help='learning rate decay (default: 0.998)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    parser.add_argument('--local_epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')


    
    parser.add_argument('--frac', type=float, default=1.0, metavar='NN',
                        help='available communication fraction each round')

    parser.add_argument('--comm_round', type=int, default=500,
                        help='how many round of communications we shoud use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')


    parser.add_argument("--tag", type=str, default="test")

    parser.add_argument("--kd_schedule", type=str, default='')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topology", type = str, default='grid')
    parser.add_argument("--rows", type = str, default=10)
    parser.add_argument("--columns", type = str, default=5)
    parser.add_argument("--type", type=str, default='epoch')

    parser.add_argument('--end_time', type=int, default=60000)

    parser.add_argument('--lambda_lr', type=float, default=0.00001)
 
    #parser.add_argument('--apply_lr_decay', default=True, action='store_true')
    parser.add_argument('--apply_lr_decay', dest='apply_lr_decay', action='store_true', help="Enable learning rate decay")
    parser.add_argument('--no_apply_lr_decay', dest='apply_lr_decay', action='store_false', help="Disable learning rate decay")
    parser.add_argument('--lr_scheduler_step_size', type=int, default=1)
    parser.add_argument('--lr_decay_rate', type=float, default=0.998)


    parser.add_argument("--n_neighbors_to_send", help="Number of neighbours to send.",type=int,
        default=2,
    )
    parser.add_argument("--masking", action='store_true')

    parser.add_argument("--uniform", action='store_true')
    parser.add_argument("--sparse", action='store_true')
    parser.add_argument("--dis_gradient_check", action='store_true')
    parser.add_argument('--dense_ratio', type=float, default=0.5,
                        help='local density ratio')
    parser.add_argument('--anneal_factor', type=float, default=0.5,
                        help='anneal factor for pruning')
    parser.add_argument("--different_initial_masks", action='store_true', help = 'different initial masks')
    parser.add_argument("--erk_power_scale", type=float, default=1 )

    parser.add_argument('--algorithm', type=str, default='blenddfl')
    

    parser.add_argument('--temperature', type=float, default=2,
                        help='softmax temperature')
    
                        
    parser.add_argument('--kd_weight', type=float, default=1.0)

    parser.add_argument('--kd_alpha', type=float, default=1.0)
    parser.add_argument('--kd_alpha_min', type=float, default=0.0)
    parser.add_argument('--kd_alpha_max', type=float, default=1.0)

    parser.add_argument('--kd_warmup', type=int, default=0)

    parser.add_argument('--end_coverage', type=float, default=0.8)
    

    parser.add_argument('--round_number_evaluation', type=int, default=10)
    parser.add_argument('--description', type=str, default='')
                        
                        
    
    
    return parser
