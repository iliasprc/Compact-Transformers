import torch
import matplotlib.pyplot as plt
import seaborn

def draw(data: torch.Tensor, name='', path=''):
    h = seaborn.heatmap(data.detach().cpu().numpy()).get_figure()
    fig = h.get_figure()
    plt.savefig(path + name + '.png',
                dpi=400)
    plt.show()