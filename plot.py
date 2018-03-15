
import postpic as pp

def field_imshow(field, ax, force_symmetric_clim=False, **kwargs):
    #fig, ax = plt.subplots()

    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'

    if force_symmetric_clim:
        c = np.max(abs(field.matrix))
        kwargs['clim'] = (-c, c)
        if 'cmap' not in kwargs:
            kwargs['cmap'] = pp.plotting.plotter_matplotlib.MatplotlibPlotter.symmap

    field = field.squeeze()

    im = ax.imshow(field.matrix.T, extent = field.extent, **kwargs)
    ax.set_xlabel('{} [{}]'.format(field.axes[0].name, field.axes[0].unit))
    ax.set_ylabel('{} [{}]'.format(field.axes[1].name, field.axes[1].unit))
    colorbar = ax.get_figure().colorbar(im, ax = ax)
    colorbar.set_label('{} [{}]'.format(field.name, field.unit))
    return im
