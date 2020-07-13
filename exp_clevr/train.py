import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import shared utils
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
from IPython import embed
from model_* import Net
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from DataLoader_2 import CLEVRDataLoader
from misc import todevice
from validate import validate
import utils
import multiprocessing
multiprocessing.set_start_method('spawn',True)

def train(args):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'question_pt': args.train_question_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.train_feature_h5,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'shuffle': True
    }
    train_loader = CLEVRDataLoader(**train_loader_kwargs)
    if args.val:
        val_loader_kwargs = {
            'question_pt': args.val_question_pt,
            'vocab_json': args.vocab_json,
            'feature_h5': args.val_feature_h5,
            'batch_size': args.batch_size,
            'num_workers': 2,
            'shuffle': False
        }
        val_loader = CLEVRDataLoader(**val_loader_kwargs)

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'vocab': train_loader.vocab,
        'dim_word': args.dim_word,
        'dim_hidden': args.hidden_size,
        'dim_vision': args.dim_vision,
        'state_size': args.state_size,
        'mid_size': args.mid_size,
        'dropout_prob': args.dropout,
        'glimpses': args.glimpses,
        'dim_edge': args.dim_edge
    }
    model_kwargs_tosave = { k:v for k,v in model_kwargs.items() if k != 'vocab' }
    model = Net(**model_kwargs)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)  # Support multiple GPUS
    else:
        model = model.to(device)
    logging.info(model)
    ################################################################

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adamax(parameters, args.lr, weight_decay=0)

    start_epoch = 0
    if args.restore:
        print("Restore checkpoint and optimizer...")
        ckpt = os.path.join(args.save_dir, 'model.pt')
        ckpt = torch.load(ckpt, map_location={'cuda:0': 'cpu'})
        start_epoch = 4
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt['state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer'])
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / args.lr_halflife))
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12, 15, 17, 19, 22], gamma=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5)
    gradual_warmup_steps = [0.25 * args.lr, 0.5 * args.lr, 0.75 * args.lr, 1.0 * args.lr]
    criterion = nn.CrossEntropyLoss().to(device)
    last_acc = 0.
    logging.info("Start training........")
    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        if epoch < len(gradual_warmup_steps):
            utils.set_lr(optimizer, gradual_warmup_steps[epoch])
        else:
            scheduler.step()
        for p in optimizer.param_groups:
            lr_rate = p['lr']
            logging.info("Learning rate: %6f" % (lr_rate))
        for i, batch in enumerate(train_loader):
            progress = epoch+i/len(train_loader)
            orig_idx, image_idx, answers, *batch_input = [todevice(x, device) for x in batch]
            batch_input = [x.detach() for x in batch_input]
            logits, loss_time = model(*batch_input)
            ##################### loss #####################
            ce_loss = criterion(logits, answers)
            loss_time = 0.01 * loss_time.mean()
            loss = ce_loss + loss_time
            ################################################
            optimizer.zero_grad()
            loss.backward() 
            nn.utils.clip_grad_value_(parameters, clip_value=0.25)
            optimizer.step()
            if (i+1) % (len(train_loader) // 20) == 0:
                logging.info("Progress %.3f  ce_loss = %.3f  time_loss = %.3f" % (progress, ce_loss.item(), loss_time.item()))
            del  answers, batch_input, logits
            torch.cuda.empty_cache()
        # save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(args.save_dir, 'model.pt')) 
        logging.info(' >>>>>> save to %s <<<<<<' % (args.save_dir))
        if args.val:
            if epoch % 1 ==0:
                valid_acc = validate(model, val_loader, device)
                logging.info('\n ~~~~~~ Valid Accuracy: %.4f ~~~~~~~\n' % valid_acc)
                if valid_acc >= last_acc:
                    last_acc = valid_acc
                    save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(args.save_dir, 'model.pt'))

def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    if torch.cuda.device_count() > 1:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    state = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
        }
    torch.save(state, filename)



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--save_dir', type=str, default='output', help='path to save checkpoints and logs')
    parser.add_argument('--input_dir', default='input')
    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--vocab_json', default='../input_human/vocab.json')
    parser.add_argument('--train_feature_h5', default='../input_all/train_features.h5')
    parser.add_argument('--val_feature_h5', default='../input_all/val_features.h5')
    parser.add_argument('--restore', action='store_true')
    # training parameters
    parser.add_argument('--lr', default=0.5e-3, type=float)
    parser.add_argument('--lr_halflife', default=50000, type=int)
    parser.add_argument('--num_epoch', default=25, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=2000, help='random seed')
    parser.add_argument('--val', action='store_true', help='whether validate after each training epoch')
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int, help='word embedding')
    parser.add_argument('--hidden_size', default=600, type=int, help='hidden state of seq2seq parser')
    parser.add_argument('--dim_vision', default=18, type=int)
    parser.add_argument('--dim_edge', default=18, type=int)
    parser.add_argument('--mid_size', default=1024, type=int)
    parser.add_argument('--state_size', default=15, type=int, help='max_num_object')
    parser.add_argument('--glimpses', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not args.restore:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
    else:
        assert os.path.isdir(args.save_dir)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    # concat obsolute path of input files
    args.train_question_pt = os.path.join(args.input_dir, args.train_question_pt)
    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
    args.train_feature_h5 = os.path.join(args.input_dir, args.train_feature_h5)
    args.val_feature_h5 = os.path.join(args.input_dir, args.val_feature_h5)
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
