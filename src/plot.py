import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

sns.set_palette('pastel')

# separate accuracy/loss curves

def make_curve(ax, name, xs, ys):
    ax.set_title(name)
    sns.lineplot(xs, ys, ax=ax)
    # ax.legend(list(y_dict.keys()))
    ax.set_ylabel('')
    plt.ylabel('')

def loss_acc(csv_file):
    df = pd.read_csv(csv_file)
    x = df.index.name = 'step'
    xs = df[x] = df.index
    fig, axes = plt.subplots(1, 2)
    (loss_ax, acc_ax) = axes
    make_curve(loss_ax, 'loss',     xs, df['loss'])
    make_curve( acc_ax, 'accuracy', xs, df['acc'])
    plt.ylabel('')
    png_file = re.sub('.csv', '.png', csv_file)
    fig.savefig(png_file)
    print(f'wrote image to {png_file}!')
    plt.close(fig)

def fix_idx(df):
    idx = df.columns[0]
    df['step'] = df[idx]
    df.index.name = 'step'
    del df[idx]
    return df

for csv_file in glob.glob(os.path.join(os.getcwd(), 'checkpoint_folder', 'baseline_*=*_*-*.csv')):
    loss_acc(csv_file)

# # attempt at combined optimizer chart

# df = fix_idx(pd.concat(map(pd.read_csv, glob.glob(os.path.join(os.getcwd(), 'results', 'mlp-pytorch-*.csv')))))
# fig, axes = plt.subplots(2, 2, figsize=(8,8))
# plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
# ((test_loss_ax, test_acc_ax), (train_loss_ax, train_acc_ax)) = axes
# sns.lineplot(x='step', y= 'test_acc' , hue='optimizer', data=df, ax=test_acc_ax)
# sns.lineplot(x='step', y= 'test_loss', hue='optimizer', data=df, ax=test_loss_ax)
# sns.lineplot(x='step', y='train_acc' , hue='optimizer', data=df, ax=train_acc_ax)
# sns.lineplot(x='step', y='train_loss', hue='optimizer', data=df, ax=train_loss_ax)
# png_path = os.path.join(os.getcwd(), 'results', 'combined.png')
# fig.savefig(png_path)

# # attempt at cat plot

# def csv_dct(csv_file):
#     df = pd.read_csv(csv_file)
#     df = fix_idx(df)
#     dct = dict(df.iloc[len(df)-1])
#     return dct

# results = list(map(csv_dct, glob.glob(os.path.join(os.getcwd(), 'results', '*.csv'))))
# df = pd.DataFrame(results)
# df_ = df[(df.algo == 'mlp') & (df.framework == 'pytorch')]

# fig = sns.catplot(
#     # hue='dnn_hidden_units',
#     x='optimizer',
#     y='test_acc',  # metric: 'test_loss', 'secs'
#     # row='batch_size',
#     # col='learning_rate',
#     # col_wrap=4,
#     data=df_[1:-1],
#     kind='boxen',
#     # split=True,
# )

# csv_path = os.path.join(os.getcwd(), 'results', 'combined.csv')
# df.to_csv(csv_path)
# png_path = os.path.join(os.getcwd(), 'results', 'chart.png')
# fig.savefig(png_path)
