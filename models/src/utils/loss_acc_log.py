import os
import csv

from src.utils.visdom_util import VisdomLinePlotter


class LossAccLogger:
    def __init__(self, path, session_name):
        self.path = os.path.join(path, 'log.csv')

        self.logs = {
            'epochs': [],
            'iteration': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        with open(self.path, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'iteration', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

        self.plotter = VisdomLinePlotter(env_name=f'Model {session_name}')

    def __call__(self, epoch, iteration, train_loss, val_loss, train_acc, val_acc):
        with open(self.path, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, iteration, train_loss, val_loss, train_acc, val_acc])

        self.logs['epochs'].append(epoch)
        self.logs['iteration'].append(iteration)
        self.logs['train_loss'].append(train_loss)
        self.logs['val_loss'].append(val_loss)
        self.logs['train_acc'].append(train_acc)
        self.logs['val_acc'].append(val_acc)

        # TODO also log epochs as line?
        self._update_plots(iteration, train_loss, val_loss, train_acc, val_acc)

    def _update_plots(self, epoch,  train_loss, val_loss, train_acc, val_acc):
        self.plotter.plot('loss', 'train', 'Loss', epoch, train_loss)
        self.plotter.plot('loss', 'val', 'Loss', epoch, val_loss)
        self.plotter.plot('acc', 'train', 'Accuracy', epoch, train_acc)
        self.plotter.plot('acc', 'val', 'Accuracy', epoch, val_acc)

    def update_plots_iterations(self, iteration, train_loss, train_acc):
        self.plotter.plot_train('loss_train', 'val', 'Iteration Loss', iteration, train_loss)
        self.plotter.plot_train('acc_train', 'train', 'Iteration Accuracy', iteration, train_acc)

    def visualize_config(self, config):
        self.plotter.plot_config(config)
