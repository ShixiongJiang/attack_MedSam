import argparse


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-encoder', type=str, default='default', help='encoder type')
    parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
    parser.add_argument('-mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument('-exp_name', default='msa_test_polyp', type=str, help='net type')
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-vis', type=int, default=1, help='visualization')
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-val_freq',type=int,default=25,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=1024, help='output_size')
    parser.add_argument('-patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=1, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-weights', type=str, default ="./checkpoint/sam/checkpoint_best.pth", help='the weights file you want to test')
    # parser.add_argument('-weights', type=str, default ="./logs/lora_baseline/Model/checkpoint_best.pth", help='the weights file you want to test')


    parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default = 0, help='the weights sim')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='Polyp' ,type=str,help='dataset name')
    # parser.add_argument('-sam_ckpt', default="./sam_checkpoints/sam_vit_b_01ec64.pth" , help='sam checkpoint address')
    parser.add_argument('-sam_ckpt', default="./sam_checkpoints/sam_vit_b_01ec64.pth" , help='sam checkpoint address')

    # parser.add_argument('-sam_ckpt', default="./logs/subpopulation_attack_0722/Model/checkpoint_best.pth" , help='sam checkpoint address')


    parser.add_argument('-thd', type=bool, default=False , help='3d or not')
    parser.add_argument('-chunk', type=int, default=96 , help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4 , help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')
    parser.add_argument('-evl_chunk', type=int, default=None , help='evaluation chunk')
    parser.add_argument('-mid_dim', type=int, default=None , help='middle dim of adapter or the rank of lora matrix')
    parser.add_argument('-epsilon', type=float, default=0.1 , help='define the power of attack')
    parser.add_argument('-attack_method', type=str, default='pgd' , help='define the power of attack')
    parser.add_argument('-freeze', type=bool, default=True, help='define the power of attack')
    parser.add_argument('--num_processes', type=int, default=4, help='Total number of processes')

    parser.add_argument("--backdoor", type=bool, default=False, help="indicate backdoor attack")
    # parser.add_argument("--poison_datasets", type=str,nargs="+",default="poison_dataset")
    parser.add_argument("--poison_datasets", type=str,nargs="+",default="cluster_poison_dataset")
    parser.add_argument("--generate_subpupu_dataset", type=str,nargs="+",default="poison_dataset")
    parser.add_argument("--generate_cluster_dataset", type=str,nargs="+",default="cluster_poison_dataset")
    parser.add_argument("--poison", type=bool, default=False, help="indicate poison attack")
    # parser.add_argument(
    # '-data_path',
    # type=str,
    # default='../../ADataset/Polyp_HSNet',
    # help='The path of segmentation data')
    parser.add_argument(
    '-data_path',
    type=str,
    default='./dataset',
    help='The path of segmentation data')

    # '../dataset/RIGA/DiscRegion'
    # '../dataset/ISIC'
    opt = parser.parse_args()

    return opt

# required=True, 
