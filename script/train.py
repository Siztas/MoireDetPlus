# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

from __future__ import print_function
import sys
sys.path.append(r"/public/home/lijinjin/下载/MoireDet++_train/MoireDet-main/MoireDet")

import os
from lib.utils import load_json
import math



config = load_json('/public/home/lijinjin/下载/MoireDet++_train/MoireDet-main/MoireDet/script/configs/10_specific_no_per_many_loss.json')



# config = load_json('./configs/test.json')


os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['trainer']['gpus']])



from lib.data_loader import get_dataloader

from lib.models import get_model,get_loss
from lib.trainer import Trainer


from torch import nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5),nonlinearity='leaky_relu') # nonlinearity must be leaky_relu
        #nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

        # nn.init.kaiming_uniform_(m.weight.data)
        # try:
        #     nn.init.kaiming_uniform_(m.bias.data)
        # except:
        #     pass

def main(config):
    train_loader = get_dataloader(config['data_loader']['type'], config['data_loader']['args'])

    criterion = get_loss(config).cuda()

    model = get_model(config)

    trainer = Trainer(config=config,
                      model=model,
                      criterion=criterion,
                      train_loader=train_loader,
                      weights_init=weights_init)
    trainer.train()


if __name__ == '__main__':
    main(config)
