
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

def plot_fields_1d(fields, ax, common_name=None, plot_method='plot', plot_args=dict(), plot_kwargs=dict()):
    for i, field in enumerate(fields):
        args = plot_args.get(i, tuple())
        kwargs = plot_kwargs.get(i, dict())
        if 'label' not in kwargs:
            kwargs['label'] = field.name
        getattr(ax, plot_method)(field.grid, field, *args, **kwargs)

    xAxis = field.axes[0]
    ax.set_xlabel('{} [{}]'.format(xAxis.name, xAxis.unit))

    yName = common_name if common_name else field.name
    ax.set_ylabel('{} [{}]'.format(yName, field.unit))
