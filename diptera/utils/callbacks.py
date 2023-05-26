from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from tensorflow.keras.callbacks import Callback
import pandas as pd
import gc

# ----


class HistoryPlot(Callback):

    def load_history(self):
        if self.logs_path.exists():  # reload history if it exists
            return pd.read_csv(self.logs_path)
        else:
            return pd.DataFrame()

    def plot(self, history, mode):
        if mode == 'loss':
            metric = self.loss_metric
            ymax = max(max(history['loss']), max(history['val_loss']))
        elif mode == 'accuracy':
            metric = self.acc_metric
            ymax = 1

        fig = plt.figure(dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(history[metric], color='r')
        ax.plot(history['val_'+metric], color='b')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(mode.capitalize())
        ax.set_ylim(0, ymax)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.legend(["Training", "Validation"], loc="lower right")
        ax.set_title(mode.capitalize() + ' history')
        fig.savefig(self.logs_path.parent / f"{mode}.png")
        plt.close(fig)
        
        gc.collect()  # fix memory leak


    def __init__(self, logs_path, acc_metric="accuracy", loss_metric="loss"):
        """
        :param logs_path:
        :param metrics: a dictionnary in the form {'categorical_accuracy': 'Accuracy'}
        """
        self.logs_path = logs_path
        self.acc_metric = acc_metric
        self.loss_metric = loss_metric

    def on_epoch_end(self, epoch, logs = None):
        logs = self.load_history()
        self.plot(logs, 'accuracy')
        self.plot(logs, 'loss')

