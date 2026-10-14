"""从已记录CSV绘图，不训练；保留每一个batch点，不做平滑或重采样。"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def plot_batches(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['path.simplify'] = False
    directory = Path(directory)
    batches = pd.read_csv(directory / 'batch_metrics.csv')
    assert batches.step.tolist() == list(range(1, len(batches) + 1))
    assert np.isfinite(batches.batch_training_objective).all()
    assert (batches.batch_training_objective > 0).all()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout='constrained')
    for ax in axes:
        ax.plot(batches.step, batches.batch_training_objective, color='#28659a', linewidth=.65)
        ax.set(xlabel='Mini-batch update step', ylabel='Training loss (including L2)', xlim=(1, len(batches)))
        ax.grid(alpha=.18)
    axes[0].set(title='Raw mini-batch loss (linear scale)', ylim=(0, None))
    axes[1].set(title='Same raw values (log scale)', yscale='log')
    fig.savefig(directory / 'batch_loss_curves.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    plot_batches(parser.parse_args().directory)
