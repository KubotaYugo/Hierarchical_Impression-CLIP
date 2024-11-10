import wandb
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def save_plot(data, y_label, filename):
    plt.plot(data)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.close()


EXP = utils.EXP
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE


wandb.login()

run_id_list = [[[1.0, 0.0, 0.0], 'yugo-kubota-kyushu-university/Hierarchical_ImpressionCLIP_5-1/p977tp0d'],
               [[1.0, 1.0, 0.0], 'yugo-kubota-kyushu-university/Hierarchical_ImpressionCLIP_5-1/8qaldj1g'],
               [[1.0, 0.0, 1.0], 'yugo-kubota-kyushu-university/Hierarchical_ImpressionCLIP_5-1/gufb41lq'],
               [[1.0, 1.0, 1.0], 'yugo-kubota-kyushu-university/Hierarchical_ImpressionCLIP_5-1/xi7po7in']]


for WEIGHTS, run_id in run_id_list:
    
    BASE_DIR = f'{EXP}/LR={LR}_BS={BATCH_SIZE}_W={WEIGHTS}'
    SAVE_DIR = f'{BASE_DIR}/plot_losses'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # wandbからデータの取得
    api = wandb.Api()
    run = api.run(run_id)
    log = run.history(keys=['loss_total', 'loss_pair', 'loss_img_epoch', 'loss_tag_epoch', 'val_loss_pair'])

    save_plot(log['loss_total'],     'loss_total_train', f'{SAVE_DIR}/loss_total_train.png')
    save_plot(log['loss_pair'],      'loss_pair_train',  f'{SAVE_DIR}/loss_pair_train.png')
    save_plot(log['loss_img_epoch'], 'loss_img_train',   f'{SAVE_DIR}/loss_img_train.png')
    save_plot(log['loss_tag_epoch'], 'loss_tag_train',   f'{SAVE_DIR}/loss_tag_train.png')
    save_plot(log['val_loss_pair'],  'loss_pair_val',    f'{SAVE_DIR}/loss_pair_val.png')



import wandb
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def save_plot(data, y_label, filename):
    plt.plot(data)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.close()


EXP = utils.EXP
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE


wandb.login()

run_id_list = [[[1.0, 0.0, 0.0], 'yugo-kubota-kyushu-university/Hierarchical_ImpressionCLIP_5-1/p977tp0d'],
               [[1.0, 1.0, 0.0], 'yugo-kubota-kyushu-university/Hierarchical_ImpressionCLIP_5-1/8qaldj1g'],
               [[1.0, 0.0, 1.0], 'yugo-kubota-kyushu-university/Hierarchical_ImpressionCLIP_5-1/gufb41lq'],
               [[1.0, 1.0, 1.0], 'yugo-kubota-kyushu-university/Hierarchical_ImpressionCLIP_5-1/xi7po7in']]


for WEIGHTS, run_id in run_id_list:
    
    BASE_DIR = f'{EXP}/LR={LR}_BS={BATCH_SIZE}_W={WEIGHTS}'
    SAVE_DIR = f'{BASE_DIR}/plot_losses'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # wandbからデータの取得
    api = wandb.Api()
    run = api.run(run_id)
    log = run.history(keys=['loss_total', 'loss_pair', 'loss_img_epoch', 'loss_tag_epoch', 'val_loss_pair'])

    save_plot(log['loss_total'],     'loss_total_train', f'{SAVE_DIR}/loss_total_train.png')
    save_plot(log['loss_pair'],      'loss_pair_train',  f'{SAVE_DIR}/loss_pair_train.png')
    save_plot(log['loss_img_epoch'], 'loss_img_train',   f'{SAVE_DIR}/loss_img_train.png')
    save_plot(log['loss_tag_epoch'], 'loss_tag_train',   f'{SAVE_DIR}/loss_tag_train.png')
    save_plot(log['val_loss_pair'],  'loss_pair_val',    f'{SAVE_DIR}/loss_pair_val.png')

    # # wandbからデータの取得
    # api = wandb.Api()
    # run = api.run(run_id)
    # log = run.history(keys=['loss_total', 'loss_pair', 'loss_img_epoch', 'loss_tag_epoch', 'val_loss_pair'
    #                         'ARR_tag2img_train', 'ARR_img2tag_train', 'ARR_tag2img_val', 'ARR_img2tag_val',
    #                         'meanARR_train', 'meanARR_val'])

    # save_plot(log['loss_total'],     'loss_total_train', f'{SAVE_DIR}/loss_total_train.png')
    # save_plot(log['loss_pair'],      'loss_pair_train',  f'{SAVE_DIR}/loss_pair_train.png')
    # save_plot(log['loss_img_epoch'], 'loss_img_train',   f'{SAVE_DIR}/loss_img_train.png')
    # save_plot(log['loss_tag_epoch'], 'loss_tag_train',   f'{SAVE_DIR}/loss_tag_train.png')
    # save_plot(log['val_loss_pair'],  'loss_pair_val',    f'{SAVE_DIR}/loss_pair_val.png')
    # save_plot(log['ARR_tag2img_train'],  'ARR_tag2img_train',    f'{SAVE_DIR}/ARR_tag2img.png')
    # save_plot(log['ARR_img2tag_train'],  'ARR_img2tag_train',    f'{SAVE_DIR}/ARR_img2tag.png')
    # save_plot(log['ARR_tag2img_val'],    'ARR_tag2img_val',    f'{SAVE_DIR}/ARR_tag2img.png')
    # save_plot(log['ARR_img2tag_val'],    'ARR_img2tag_val',    f'{SAVE_DIR}/ARR_img2tag.png')
    # save_plot(log['meanARR_train'],      'meanARR_train',        f'{SAVE_DIR}/meanARR.png')
    # save_plot(log['meanARR_val'],        'meanARR_val',        f'{SAVE_DIR}/meanARR.png')