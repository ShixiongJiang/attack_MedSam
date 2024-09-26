# train_reverse_adaptation.py
from dataset import *
from utils import *
import function_r as function
from torch.utils.data  import  DataLoader
from collections import OrderedDict
import torchvision.transforms as transforms

args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
import warnings
warnings.filterwarnings("ignore")
'''load pretrained model'''
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
# print(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
loc = 'cuda:{}'.format(args.gpu_device)
checkpoint = torch.load(checkpoint_file, map_location=loc)
start_epoch = checkpoint['epoch']
best_tol = checkpoint['best_tol']
state_dict = checkpoint['state_dict']
if args.distributed != 'none':
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict)

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.ToTensor()
])


temp_data_path=args.data_path
# for  dataset  in  ["CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB", "Kvasir", "CVC-300"] :
for  dataset  in  ["CVC-ClinicDB"] :

    args.data_path=os.path.join(temp_data_path,"TestDataset",dataset)
    polyp_test_dataset = Polyp2(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')
    nice_test_loader = DataLoader(polyp_test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, start_epoch, net)
    logger.info(f'total loss: {tol} IOU: {eiou}, DICE: {edice} ')



