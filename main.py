import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.resnet as module_arch_resnet
import model.swin_transformer_1d as module_arch_swin_transformer_1d
import model.beat_aligned_transformer as module_arch_beat_aligned_transformer

from parse_config import ConfigParser
from trainer import Trainer
from evaluater import Evaluater
from model.metric import ChallengeMetric
from utils.util import load_model
from utils.lr_scheduler import CosineAnnealingWarmUpRestarts, GradualWarmupScheduler
import datetime

files_models = {
    "resnet": ['resnet'],
    "swin_transformer_1d": ['swin_transformer'],
    "beat_aligned_transformer": ['beat_aligned_transformer']
}

def main(config):
    logger = config.get_logger('train')


    # build model architecture, then print to console
    global model
    for file, types in files_models.items():
        for type in types:
            if config["arch"]["type"] == type:
                model = config.init_obj('arch', eval("module_arch_" + file))
                logger.info(model)
                if config['arch'].get('weight_path', False):
                    model = load_model(model, config["arch"]["weight_path"])

    criterion = getattr(module_loss, config['loss']['type'])

    # get function handles of metrics

    challenge_metrics = ChallengeMetric(config['data_loader']['args']['label_dir'])
    # challenge_metrics = ChallengeMetric2(num_classes=9)

    metrics = [getattr(challenge_metrics, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    if config["lr_scheduler"]["type"] == "CosineAnnealingWarmRestarts":
        params = config["lr_scheduler"]["args"]
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=params["T_0"], T_mult=params["T_mult"],
                                                     T_up=params["T_up"], gamma=params["gamma"], eta_max=params["eta_max"])
    elif config["lr_scheduler"]["type"] == "GradualWarmupScheduler":
        params = config["lr_scheduler"]["args"]
        scheduler_steplr_args = dict(params["after_scheduler"]["args"])
        scheduler_steplr = getattr(torch.optim.lr_scheduler, params["after_scheduler"]["type"])(optimizer, **scheduler_steplr_args)
        lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=params["multiplier"], total_epoch=params["total_epoch"], after_scheduler=scheduler_steplr)
    else:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    if config["only_test"] == False:
        # setup data_loader instances
        data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.valid_data_loader
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)
        trainer.train()

    evaluater = Evaluater(model, criterion, metrics,
                          config=config)
    evaluater.evaluate()

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--seed', type=int, default=0)
    args.add_argument('-t', '--only_test', default=False, type=bool,
                      help='only test (default: False)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    import os
    print("torch.cuda.device_count(): ", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    main(config)

    end_time = datetime.datetime.now()
    print("程序运行时间：" + str((end_time - start_time).seconds) + "秒")
