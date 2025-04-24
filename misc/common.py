# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:33:19 2024

@author: renxi
"""
import numpy as np
import matplotlib.pylab as plt

def log_csv(train_loss, val_loss, train_score, val_score, fpath):
    with open(fpath, 'w') as fo:
        print('train_loss, val_loss, train_score, val_score', file=fo)
        for l1, l2, s1, s2 in zip(train_loss, val_loss, train_score, val_score):
            print('%.3f, %.3f, %3f, %3f' % (l1,l2,s1,s2), file=fo)


def plot_train_val_info(data, des=[], xlab='Epoch', ylab='Loss', 
                        title = 'Train and val Loss', save_path = ''):            
    plt.style.use("ggplot")
    plt.figure()
    
    n_curs = len(data)
    for idx in range(n_curs):
        dt = data[idx]
        if idx < len(des):
            label = des[idx]
        else:
            label = 'untitled'    
        plt.plot(dt, label=label)
        
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc="lower left")
    plt.savefig(save_path, dpi=200)            
    
    
if __name__ == '__main__':
    data = [[1,2,3,4,5,5],[34,12,32,41,12,22]]
    des=['per1','per2']
    plot_train_val_info(data, des=des, xlab='epoch', ylab='loss', save_path='fig.png')     