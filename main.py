import os
import argparse
from solver import Solver
from torch.backends import cudnn

def str2bool(v):
    return v.lower() == 'true'

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    os.system('cp -r models %s/'%config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.latest_path):
        os.makedirs(config.latest_path)        

    # Solver
    solver = Solver(config)
    solver.train()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--obj', type=str)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--random', type=str2bool, default=False)

    # model
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lambda_constrain', type=float, default=2)
    parser.add_argument('--lambda_triplet', type=float, default=0)
    parser.add_argument('--gen_triplet', type=str2bool, default=False)

    # ablation stud for rebuttal
    parser.add_argument('--ablation_recon_kl', type=str2bool, default=True)
    parser.add_argument('--ablation_const_l2', type=str2bool, default=True)
    parser.add_argument('--ablation_const_angle', type=str2bool, default=True)
    parser.add_argument('--ablation_tri', type=str, default='max,mean')

    # optimizor
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # training procedure
    parser.add_argument('--d_train_repeat', type=int, default=5)
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # log
    parser.add_argument('--root_path', type=str, default='expr')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--flag', type=str, default='scde')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()

    config.ablation_tri = config.ablation_tri.split(',')
    config.log_path = os.path.join(config.root_path, config.log_path)
    config.model_save_path = os.path.join(config.root_path, config.model_save_path)
    config.sample_path = os.path.join(config.root_path, config.sample_path)
    config.log_file = os.path.join(config.log_path, config.flag+'.txt')
    config.latest_path = os.path.join(os.environ['HOME'], 'remote', 'latest_res', config.root_path.split('/')[-1])
    
    print(config)
    main(config)
