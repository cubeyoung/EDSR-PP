import os
import math
import time
import datetime
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import skimage.io as sio
import skimage.color as sc

from model import model
from loss import loss

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as tu
from torch.autograd import Variable

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '/data01/kky/experiment/' + args.save
        else:
            self.dir = '/data01/kky/experiment/' + args.load
            if not os.path.exists(self.dir): args.load = '.'

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def load(self):
        my_model = model(self.args).get_model()
        trainable = filter(lambda x: x.requires_grad, my_model.parameters())

        if self.args.optimizer == 'SGD':
            optimizer_function = optim.SGD
            kwargs = {'momentum': self.args.momentum}
        elif self.args.optimizer == 'ADAM':
            optimizer_function = optim.Adam
            kwargs = {
                'betas': (self.args.beta1, self.args.beta2),
                'eps': self.args.epsilon}
        elif self.args.optimizer == 'RMSprop':
            optimizer_function = optim.RMSprop
            kwargs = {'eps': self.args.epsilon}

        kwargs['lr'] = self.args.lr
        kwargs['weight_decay'] = 0
        my_optimizer = optimizer_function(trainable, **kwargs)

        if self.args.decay_type == 'step':
            my_scheduler = lrs.StepLR(
                my_optimizer,
                step_size=self.args.lr_decay,
                gamma=self.args.gamma)

        elif self.args.decay_type.find('step') >= 0:
            milestones = self.args.decay_type.split('_')
            milestones.pop(0)
            milestones = list(map(lambda x: int(x), milestones))
            my_scheduler = lrs.MultiStepLR(
                my_optimizer,
                milestones=milestones,
                gamma=self.args.gamma)

        self.log_training = torch.Tensor()
        self.log_test = torch.Tensor()
        if self.args.load == '.':
            my_loss = loss(self.args).get_loss()
        else:
            if not self.args.test_only:
                self.log_training = torch.load(self.dir + '/log_training.pt')
                self.log_test = torch.load(self.dir + '/log_test.pt')

            resume = self.args.resume
            if resume == -1:
                my_model.load_state_dict(
                    torch.load(self.dir + '/model/model_lastest.pt'))
                resume = len(self.log_test)
            else:
                my_model.load_state_dict(
                    torch.load(self.dir + '/model/model_{}.pt'.format(resume)))

            my_loss = torch.load(self.dir + '/loss.pt')
            my_optimizer.load_state_dict(
                torch.load(self.dir + '/optimizer.pt'))

            print('Load loss function from checkpoint...')
            print('Continue from epoch {}...'.format(resume))

        return my_model, my_loss, my_optimizer, my_scheduler

    def add_log(self, log, train=True):
        if train:
            self.log_training = torch.cat([self.log_training, log])
        else:
            self.log_test = torch.cat([self.log_test, log])

    def save(self, trainer, epoch):
        torch.save(
            trainer.model.state_dict(),
            self.dir + '/model/model_lastest.pt')
        if not self.args.test_only:
            torch.save(
                trainer.model.state_dict(),
                '{}/model/model_{}.pt'.format(self.dir, epoch))
            torch.save(
                trainer.loss,
                self.dir + '/loss.pt')
            torch.save(
                trainer.optimizer.state_dict(),
                self.dir + '/optimizer.pt')
            torch.save(
                self.log_training,
                self.dir + '/log_training.pt')
            torch.save(
                self.log_test,
                self.dir + '/log_test.pt')
            self.plot(trainer, epoch, self.log_training, self.log_test)

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot(self, trainer, epoch, training, test):
        axis = np.linspace(1, epoch, epoch)

        def _init_figure(label):
            fig = plt.figure()
            plt.title(label)
            plt.xlabel('Epochs')
            plt.grid(True)

            return fig
           
        def _close_figure(fig, filename):
            plt.savefig(filename)
            plt.close(fig)

        for i, loss in enumerate(trainer.loss):
            label = '{} Loss'.format(loss['type'])
            fig = _init_figure(label)
            plt.plot(axis, training[:, i].numpy(), label=label)
            plt.legend()
            _close_figure(fig, '{}/loss_{}.pdf'.format(self.dir, loss['type']))

        set_name = type(trainer.loader_test.dataset).__name__
        fig = _init_figure('SR on {}'.format(set_name))
        for idx_scale, scale in enumerate(self.args.scale):
            legend = 'Scale {}'.format(scale)
            plt.plot(axis, test[:, idx_scale].numpy(), label=legend)
            plt.legend()

        _close_figure(
            fig,
            '{}/test_{}.pdf'.format(self.dir, set_name))

    def save_results(self, idx, input, output, target, scale):
        if self.args.save_results:
  
            dir = '/home/kky/EDSR-PP-Pytorch-master/Results'

            filename = '%s/%04dx%d'%(dir, idx + 1, 4)
            v = output
            n = 'EDSR-PP'
            tu.save_image(
                    v.data[0] / self.args.rgb_range,
                    '{}{}.png'.format(filename,n))

def chop_forward(x, model, scale, shave=10, min_size=160000, nGPUs = 1):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        
        outputlist = []
        for i in range(0, 4, nGPUs):
            input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            output_batch = model(input_batch).data[0]
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale
    
    output = Variable(x.data.new(b, c, h, w))
    
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()
        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda(0)

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        return Variable(ret, volatile=v.volatile)

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [chop_forward(aug,model,8) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')
    
    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output

def quantize(img, rgb_range):
    return img.mul(255 / rgb_range).clamp(0, 255).add(0.5).floor().div(255)

def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0)
    yCbCr = sc.rgb2ycbcr(rgb) / 255

    return torch.Tensor(yCbCr[:, :, 0])

def calc_PSNR(input, target, set_name, rgb_range, scale):
    # We will evaluate these datasets in y channel only
    test_Y = ['Set5', 'Set14', 'B100', 'Urban100']
    (_, c, h, w) = input.size()
    input = quantize(input.data[0], rgb_range)
    target = quantize(target[:, :, 0:h, 0:w].data[0], rgb_range)
    diff = input - target
    if set_name in test_Y:
        shave = scale
        if c > 1:
            input_Y = rgb2ycbcrT(input.cpu())
            target_Y = rgb2ycbcrT(target.cpu())
            diff = (input_Y - target_Y).view(1, h, w)
    else:
        shave = scale + 6

    diff = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diff.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr

