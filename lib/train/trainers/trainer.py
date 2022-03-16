import time
import datetime
import pandas as pd
import torch
import tqdm
from torch.nn import DataParallel
import os
import numpy as np
import torchsnooper

 
# 自己添加的loss           
def distence_loss(vertex_pred, vertex_true, vertex_weights): 

    b, vertex_dim, h, w =vertex_pred.shape
    dx = vertex_pred[:, 0::2]
    #print('dx.shape',dx.shape)
    dy = vertex_pred[:, 1::2]
    vertex_norm = torch.sqrt(dx*dx + dy*dy + 1e-8)
    dx = dx/(vertex_norm+1e-4)
    dy = dy/(vertex_norm+1e-4)
    #print("dx",dx.shape)
    gx = vertex_true[:, 0::2]
    gy = vertex_true[:, 1::2]
    dist_loss = torch.sqrt(((gx - dx) ** 2) +((gy - dy) ** 2))*vertex_weights
    sigma = 1.0
    beta = 1. / (sigma ** 2)
    smoothL1_sign = (dist_loss < beta)
    dist_loss = torch.where(smoothL1_sign, 0.5 * (sigma*dist_loss) ** 2, dist_loss - 0.5 / (sigma **2))
    dist_loss = torch.mean(torch.sum(dist_loss.view(b, -1), dim=1) / (vertex_dim/2 * torch.sum(vertex_weights.view(b,-1), dim=1) + 1e-8) )
    #print('dist_loss',dist_loss)
    return dist_loss 


class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        network = DataParallel(network)
        self.network = network
    
    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch
    #@torchsnooper.snoop() 
    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1
            batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats, weight = self.network(batch)


            dist_loss = distence_loss(output['vertex'], batch['vertex'], weight)
            loss += dist_loss * 0.01 * np.min([1.5 ** epoch, 10])


            loss_stats.update({'dist_loss': dist_loss, 'loss':loss})    
            # training stage: loss; optimizer; scheduler
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)
            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            with torch.no_grad():
                output, loss, loss_stats, image_stats, _ = self.network.module(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
            

