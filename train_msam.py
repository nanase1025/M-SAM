# set up environment
from ast import Param
from inspect import Parameter
from sqlite3 import adapters
import numpy as np
import random 
import datetime
import logging
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.new_build_sam3D import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader, Dataset_Union_ALL_Test, Union_Dataloader_Test
from utils.data_paths import img_datas
from torch.utils.data import random_split
from Unet import UNet3D
from sam_lora_image_encoder import LoRA_Sam
from mea import BiDirectionalAttentionNetwork
# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='testtest111')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='/home/featurize/work/miccai_project/SAM-Med3D/work_dir/SAM/sam_med3d_turbo.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir')

# train
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

class CombinedModel(nn.Module):
    def __init__(self, lora_sam, bidatt):
        super(CombinedModel, self).__init__()
        self.lora_sam = lora_sam
        self.bidatt = bidatt

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
click_methods = {
    'random': get_next_click3D_torch_2,
}

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    return sam_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(paths=img_datas, transform=tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.img_size,args.img_size,args.img_size)),
        tio.RandomFlip(axes=(0, 1, 2)),
    ]),
    threshold=1000)

    # 划分数据集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    print(len(train_dataset))

    # 训练集Sampler
    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # 训练集DataLoader
    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 验证集DataLoader（不需要Sampler）
    val_dataloader = Union_Dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

class BaseTrainer:
    def __init__(self, lora_bid_sam, train_dataloader, val_dataloader, args, loggers):

        self.model = lora_bid_sam
        self.dataloaders = train_dataloader
        self.val_dataloader = val_dataloader
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_val_loss = np.inf
        self.best_val_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.val_dices = []
        self.val_losses = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        # self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        self.start_epoch = 0
        self.loggers = loggers
        
    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    
    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model.lora_sam

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.sam.image_encoder.parameters()}, # , 'lr': self.args.lr * 0.1},
            {'params': sam_model.sam.prompt_encoder.parameters() , 'lr': self.args.lr * 0.1},
            {'params': sam_model.sam.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
            {'params': self.model.bidatt.parameters(),'lr': self.args.lr},
        ], lr=self.args.lr, betas=(0.9,0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))
    
    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        # torch.Size([2, 2, 384]) torch.Size([2, 384, 8, 8, 8])

        low_res_masks,_ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            adapter = self.model.bidatt
        )
        
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4,args.img_size//4))
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)
            low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=[points_input, labels_input])
            return_loss += self.seg_loss(prev_masks, gt3D)
        return prev_masks, return_loss
    
    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)
            
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2*volume_intersect / volume_sum
    
        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list)/len(dice_list)).item() 


    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        total_dice = 0
        total_batches = 0

        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model.lora_sam.sam
            self.args.rank = -1

        tbar = tqdm(self.dataloaders) if not self.args.multi_gpu or self.args.rank == 0 else self.dataloaders
        
        for _, (image3D, gt3D) in enumerate(tbar):
            image3D = self.norm_transform(image3D.squeeze(dim=1)).unsqueeze(dim=1)
            image3D, gt3D = image3D.to(device), gt3D.to(device).type(torch.long)  
            
            image_embedding = sam_model.image_encoder(image3D)

            self.click_points = []
            self.click_labels = []

            pred_list = []

            prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=11) 
            
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_dice = self.get_dice_score(prev_masks, gt3D)
            total_dice += batch_dice
            total_batches += 1

        epoch_loss /= total_batches if total_batches > 0 else 1
        average_dice = total_dice / total_batches if total_batches > 0 else 0

        return epoch_loss, epoch_iou, average_dice

    def eval_epoch(self, epoch, num_clicks):
        return 0
    
    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()
    
    def validate_epoch(self):
        self.model.eval()
        sam_model = self.model.lora_sam.sam
        val_loss = 0.0
        val_dice = 0.0
        total_batches = 0
        with torch.no_grad():
            for image3D, gt3D in self.val_dataloader:
                image3D = self.norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)
                
                image3D = image3D.to(device) # batch, 1, 128, 128, 128
                gt3D = gt3D.to(device).type(torch.long)

                image_embedding = sam_model.image_encoder(image3D)

                self.click_points = []
                self.click_labels = []

                prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=11) 

                val_loss += loss.item()

                dice_score = self.get_dice_score(prev_masks, gt3D)
                val_dice += dice_score
                
                total_batches += 1

        avg_val_loss = val_loss / total_batches
        avg_val_dice = val_dice / total_batches
        return avg_val_loss, avg_val_dice


    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.loggers.info(f'Epoch: {epoch + 1}/{self.args.num_epochs}')

            num_clicks = np.random.randint(1, 21)

            epoch_loss, epoch_iou, epoch_dice = self.train_epoch(epoch, num_clicks)

            val_loss, val_dice = self.validate_epoch()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):

                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                self.val_dices.append(val_dice)
                self.val_losses.append(val_loss)
                self.loggers.info(f'Epoch {epoch + 1} - Train Loss: {epoch_loss}, Train Dice: {epoch_dice}, Val Loss: {val_loss}, Val Dice: {val_dice}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                self.save_checkpoint(epoch, state_dict, describe='latest')

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, state_dict, describe='loss_best')
                    print("save best loss ckpt!")

                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')
                    print("save best val ckpt!")

                self.plot_result(self.val_losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.val_dices, 'Dice', 'Dice')

        self.loggers.info('=====================================================================')
        self.loggers.info(f'Best Train Loss: {self.best_loss}')
        self.loggers.info(f'Best Train Dice: {self.best_dice}')
        self.loggers.info(f'Best Val Loss: {self.best_val_loss}')
        self.loggers.info(f'Best Val Dice: {self.best_val_dice}')
        self.loggers.info(f'Total Train Loss: {self.losses}')
        self.loggers.info(f'Total Train Dice: {self.dices}')
        self.loggers.info(f'Total Val Loss: {self.val_losses}')
        self.loggers.info(f'Total Val Dice: {self.val_dices}')
        self.loggers.info('=====================================================================')
        self.loggers.info(f'args: {self.args}')
        self.loggers.info(f'Used datasets: {img_datas}')
        self.loggers.info('=====================================================================')

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        
def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Load datasets
        dataloaders, val_dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        last_ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(last_ckpt['model_state_dict'])
        model.to(device)
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        lora_sam = LoRA_Sam(model, 4).to(device)
        
        batch_size, channels, depth, height, width = args.batch_size, 384, 8, 8, 8
        bidatt = BiDirectionalAttentionNetwork(channels, (depth, height, width), num_attention_blocks=1).to(device)
        
        lora_bid_sam = CombinedModel(lora_sam, bidatt)
        # print(lora_bid_sam)

        print(print_number_of_trainable_model_parameters(lora_bid_sam))
        loggers = get_logger(os.path.join(LOG_OUT_DIR, "logs", f"{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
        # Create trainer
        trainer = BaseTrainer(lora_bid_sam, dataloaders, val_dataloaders, args, loggers)
        # Train
        trainer.train()

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
