import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from PIL import Image


def gibbs_sampling(mean, cov, sample_size):
    '''
    param : 
        mean : np.ndarray 平均値
        cov : np.ndarray 共分散
        sample_size : int サンプリングする数

    return:
        rtype: np.ndarray
    '''
    samples = []
    n_dim = mean.shape[0]
    origin = [0, 0]
    samples.append(origin)
    search_dim = 0

    for _ in range(sample_size):
        if search_dim == n_dim - 1:
            search_dim = 0
        else:
            search_dim = search_dim + 1

        prev_sample = samples[-1][:]
        A = cov[search_dim][search_dim - 1] / float(cov[search_dim - 1][search_dim - 1])
        _y = prev_sample[search_dim - 1]

        _mean = mean[search_dim] + A * (_y - mean[search_dim - 1])
        sigma_zz = cov[search_dim][search_dim] - A * cov[search_dim - 1][search_dim]

        sample_x = np.random.normal(loc=_mean, scale=np.power(sigma_zz, .5), size=1)
        prev_sample[search_dim] = sample_x[0]
        samples.append(prev_sample)
    return np.array(samples)


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def make_gif():
    '''gifの作成'''
    # 出力先のディレクトリを作成
    if not os.path.exists('result'):
        os.mkdir('result')
    files = sorted(glob.glob('gibbs_sampling_plot/gibbs_sampling*.png'), key=numerical_sort)
    images = list(map(lambda file: Image.open(file), files))
    images[0].save('result/gibbs_sampling.gif', save_all=True, append_images=images[1:], duration=1000, loop=0)


def line_plotter(ax, data_x, data_y, param_dict):
    return ax.plot(data_x, data_y, **param_dict)


if __name__ == '__main__':
    # 出力先のディレクトリを作成
    if not os.path.exists('gibbs_sampling_plot'):
        os.mkdir('gibbs_sampling_plot')
    # サンプルサイズ
    sample_size = 100 + 1

    # 2次元正規分布
    n_dim = 2
    mean = np.ones(n_dim)
    covariance = np.array([[3, 0.5], [0.5, 3]])

    eig_values, _ = np.linalg.eig(covariance)
    avg_eigs = np.average(eig_values)
    sample = gibbs_sampling(mean, covariance, sample_size)

    # 2次元ガウス分布の確率密度関数を計算
    multi_norm = stats.multivariate_normal(mean=mean, cov=covariance)
    X, Y = np.meshgrid(np.linspace(mean[0] - avg_eigs * 2, mean[0] + avg_eigs * 2, sample_size),
                       np.linspace(mean[1] - avg_eigs * 2, mean[1] + avg_eigs * 2, sample_size))

    fig, ax = plt.subplots(figsize=(4, 4))
    # scipyのお作法--多次元正規分布--
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
    Pos = np.empty(X.shape + (2,))
    Pos[:, :, 0] = X
    Pos[:, :, 1] = Y
    Z = multi_norm.pdf(Pos)

    # 等高線
    ax.contour(X, Y, Z, colors='C0')
    ax.scatter(sample[0, 0], sample[0, 1], marker='o', alpha=1., s=30., edgecolor='C1')

    # ギブスサンプリングの性質の可視化
    # ①x方向へ動くときはyを固定した条件付き確率に従う。
    # ②y方向へ動くときはxを固定した条件付き確率に従う。
    for i in range(1, sample_size):
        if i == 1:
            ax.scatter(sample[0:i, 0], sample[0:i, 1], marker='o', alpha=1., s=30., edgecolor='C1', label='Samples')
            line_plotter(ax, [sample[i-1, 0], sample[i, 0]], [sample[i-1, 1], sample[i, 1]], {'marker': 'o', 'color': 'C1'})
        else:
            ax.scatter(sample[0:i, 0], sample[0:i, 1], marker='o', alpha=1., s=30., edgecolor='C1')
            line_plotter(ax, [sample[i-1, 0], sample[i, 0]], [sample[i-1, 1], sample[i, 1]], {'marker': 'o', 'color': 'C1'})
        ax.legend()
        ax.set_title('Gibbs Sampling')
        fig.tight_layout()
        fig.savefig('gibbs_sampling_plot/gibbs_sampling'+str(i)+'.png', dpi=150)

    # animationの作成
    make_gif()
