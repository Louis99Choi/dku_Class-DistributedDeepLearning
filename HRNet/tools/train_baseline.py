# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchviz import make_dot
import matplotlib.pyplot as plt

# for pruning & quantization
from torch.nn.utils import prune
import torch.quantization as tq

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils, pytorchtools

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# early_stopping = pytorchtools.EarlyStopping(patience = 10, verbose = True)

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename(ex_ face_alignment_rat-prune-0.2_hrnet_w18.yaml)',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args

# Model visualization
def visualize_model_structure(model, file_name="baseline_structure"):
    """
    Model visualization and save the structure to a file
    """
    dummy_input = torch.randn(1, 3, 256, 256)
    model_dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    model_dot.render(file_name, format="png")

# Model Pruning 
def apply_pruning(model, amount):
    """
    Apply pruning to Conv2d layers in the model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Pruning each Channel by amount
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=1)
            prune.remove(module, "weight")  # Remove pruning mask to restore tensor structure
            
def apply_quantization(model):
    """
    Apply Post-Training Quantization to the model.
    """
    model_int8 = tq.quantize_dynamic(model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)
    return model_int8
            

def main():
    ### Base directory for model structures
    base_dir = "./model_structures/"
    
    args = parse_args()
    
    #######
    # print(os.path.splitext(os.path.basename(args.cfg))[0]) # face_alignment_rat-base_hrnet_w18
    #######

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_face_alignment_net(config)
    
    ### Apply Quantization if enabled
    if config.QUANTIZATION.QUANTIZE:
        print(f"Applying quantization with type: {config.QUANTIZATION.QUANTIZE_TYPE}")
        model = apply_quantization(model)
    
    ### Apply Pruning if enabled
    if config.PRUNING.PRUNE:
        print(f"Applying pruning with amount: {config.PRUNING.AMOUNT}")
        apply_pruning(model, 0.5)
    
    # ### Apply Quantization if enabled
    # if config.QUANTIZATION.QUANTIZE:
    #     print(f"Applying quantization with type: {config.QUANTIZATION.QUANTIZE_TYPE}")
    #     model = apply_quantization(model)
    
    ### Model visualization and save the structure to a file
    print(f"Visualizing model structure[{os.path.splitext(os.path.basename(args.cfg))[0]}]...")
    # print(model)
    
    
    visualize_model_structure(model, file_name=f"{base_dir}/{os.path.splitext(os.path.basename(args.cfg))[0]}")

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # loss
    criterion = torch.nn.MSELoss(reduction='mean').cuda()

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    dataset_type = get_dataset(config)

    train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=True,is_test=False),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    val_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False,is_test=False),
        batch_size=config.VALID.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)
        lr_scheduler.step()
        # evaluate
        nme, predictions = function.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)

        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)
        utils.save_checkpoint(
            {"state_dict": model,
             "epoch": epoch + 1,
             "best_nme": best_nme,
             "optimizer": optimizer.state_dict(),
             }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))
        # early_stopping(nme,model)

        # if early_stopping.early_stop:
        #     print('end the training by early stop')
        #     break

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
    
    # weight distribution visualization
    weights = [param.detach().cpu().numpy().flatten() for name, param in model.named_parameters()]
    plt.hist(weights, bins=100)
    plt.title("Baseline Weight Distribution")
    plt.savefig("baseline_weight_distribution.png")
    plt.show()


if __name__ == '__main__':
    main()










