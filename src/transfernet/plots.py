from matplotlib import pyplot as pl
from sklearn import metrics

import matplotlib
import json

# Font styles
font = {'font.size': 16, 'lines.markersize': 10}
matplotlib.rcParams.update(font)


def parity_plotter(y, y_pred, sigma_y, save, group):

    '''
    Make a paroody plot.

    inputs:
        save = The directory to save plot.
    '''

    rmse = metrics.mean_squared_error(y, y_pred)**0.5

    if y.shape[0] > 1:
        rmse_sigma = rmse/sigma_y
    else:
        rmse_sigma = np.nan

    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)

    label = r'$RMSE/\sigma_{y}=$'
    label += r'{:.2}'.format(rmse_sigma)
    label += '\n'
    label += r'$RMSE=$'
    label += r'{:.2}'.format(rmse)
    label += '\n'
    label += r'$MAE=$'
    label += r'{:.2}'.format(mae)
    label += '\n'
    label += r'$R^{2}=$'
    label += r'{:.2}'.format(r2)

    fig, ax = pl.subplots()

    if group == 'train':
        color = 'g'
    elif group == 'validation':
        color = 'b'
    elif group == 'test':
        color = 'r'

    ax.scatter(
               y,
               y_pred,
               marker='.',
               zorder=2,
               color=color,
               label=label,
               )

    limits = []
    min_range = min(min(y), min(y_pred))
    max_range = max(max(y), max(y_pred))
    span = max_range-min_range
    limits.append(min_range-0.1*span)
    limits.append(max_range+0.1*span)

    # Line of best fit
    ax.plot(
            limits,
            limits,
            label=r'$y=\hat{y}$',
            color='k',
            linestyle=':',
            zorder=1
            )

    ax.set_aspect('equal')
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_ylabel(r'$\hat{y}$')
    ax.set_xlabel('y')

    h = 8
    w = 8

    fig.set_size_inches(h, w, forward=True)
    fig.tight_layout()

    fig_legend, ax_legend = pl.subplots()
    ax_legend.axis(False)
    legend = ax_legend.legend(
                              *ax.get_legend_handles_labels(),
                              frameon=False,
                              loc='center',
                              bbox_to_anchor=(0.5, 0.5)
                              )
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)

    fig.savefig(save, bbox_inches='tight')
    fig_legend.savefig(
                       save.replace('.png', '_legend.png'),
                       bbox_inches='tight',
                       )

    pl.close(fig)
    pl.close(fig_legend)

    data = {}
    data[r'$RMSE$'] = float(rmse)
    data[r'$RMSE/\sigma_{y}$'] = float(rmse_sigma)
    data[r'$MAE$'] = float(mae)
    data[r'$R^{2}$'] = float(r2)
    data['y'] = y.tolist()
    data['y_pred'] = y_pred.tolist()

    jsonfile = save.replace('.png', '.json')
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def parity(
           df,
           save='.',
           ):

    for group, values in df.groupby('set'):

        y = values['y']
        sigma_y = y.std()
        y = y.values
        y_pred = values['y_pred'].values

        newsave = save+'_{}.png'.format(group)

        parity_plotter(y, y_pred, sigma_y, newsave, group)


def learning_curve(df, save_dir):

    for group, values in df.groupby('set'):

        if group == 'train':
            color = 'g'
        elif group == 'validation':
            color = 'b'
        elif group == 'test':
            color = 'r'

        # Regular plot
        fig, ax = pl.subplots()

        x = values['epoch'].values
        y = values['mae'].values

        val = min(y)

        label = '{}: lowest MAE value: {:.2f}'.format(group.capitalize(), val)
        label += '\n'
        label += '{}: last MAE value: {:.2f}'.format(group.capitalize(), y[-1])

        ax.plot(
                values['epoch'],
                values['mae'],
                marker='o',
                color=color,
                label=label,
                )

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss Mean Average Error')

        fig.tight_layout()
        name = save_dir+'_{}'.format(group)
        fig.savefig(
                    name+'.png',
                    bbox_inches='tight',
                    )

        # Legend by itself
        fig_legend, ax_legend = pl.subplots()
        ax_legend.axis(False)
        legend = ax_legend.legend(
                                  *ax.get_legend_handles_labels(),
                                  frameon=False,
                                  loc='center',
                                  bbox_to_anchor=(0.5, 0.5)
                                  )
        ax_legend.spines['top'].set_visible(False)
        ax_legend.spines['bottom'].set_visible(False)
        ax_legend.spines['left'].set_visible(False)
        ax_legend.spines['right'].set_visible(False)

        fig_legend.savefig(
                           name+'_legend.png',
                           bbox_inches='tight',
                           )

        data = {}
        data['mae'] = values['mae'].tolist()
        data['epoch'] = values['epoch'].tolist()

        with open(name+'.json', 'w') as handle:
            json.dump(data, handle)
