from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from tensorflow.keras.callbacks import Callback
import os
import pandas as pd

# ----


class PlotHistory(Callback):

    def load_history(self):
        if os.path.exists(self.logs_path):  # reload history if it exists
            return pd.read_csv(self.logs_path, header=None, index_col=0).squeeze("columns").to_dict()
        else:
            return {}

    def save_plot(self, history, key, label):

        plt.figure(dpi=120)
        fig, ax = plt.subplots(1, 1)
        ax.plot(history[key], color='r')
        ax.plot(history['val_' + key], color='b')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.legend(["Training", "Validation"], loc="lower right")
        ax.set_title(f'{label.capitalize()} history')
        fig.savefig(os.path.join(dir, label+'.png'))
        plt.close()

    def __init__(self, logs_path, metrics):
        """
        :param logs_path:
        :param metrics: a dictionnary in the form {'categorical_accuracy': 'Accuracy'}
        """
        self.logs_path = logs_path
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs = None):
        logs = self.load_history()
        for key, label in self.metrics.items():
            self.save_plot(logs, key, label)

