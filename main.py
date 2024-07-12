from train_mome import MoMEModel
import torch
import random
import numpy as np

def set_random_seed(seed):
    print("* random_seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='AliExpress_NL', choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--num_task', type=int, default=2)
    parser.add_argument('--num_expert', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--expert_layer_dims', nargs='+', type=int, default=[512,256])
    parser.add_argument('--M', nargs='+', type=int, default=10)
    parser.add_argument('--lamba_1', nargs='+', type=float, default=1e-6)
    parser.add_argument('--lamba_2', nargs='+', type=float, default=1e-6)
    args = parser.parse_args()
    set_random_seed(30)

    model = MoMEModel(data_name=args.dataset_name,
                    epochs=args.epoch,
                    num_task=args.num_task,
                    num_expert=args.num_expert,
                    lr=args.learning_rate,
                    batch_size=args.batch_size,
                    weight_decay=args.weight_decay,
                    expert_layer_dims=args.expert_layer_dims,
                    tower_layer_dims=args.tower_dims,
                    M=args.M,
                    lamba_1=args.lamba_1,
                    lamba_2=args.lamba_2
                    )
    model.train()

