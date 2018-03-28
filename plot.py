
import matplotlib
import postpic as pp
import numpy as np

def field_imshow(field, ax, force_symmetric_clim=False, log10plot=False, **kwargs):
    if log10plot:
        clim = kwargs.pop('clim', (None, None))
        kwargs['norm'] = matplotlib.colors.LogNorm(*clim)
        if clim[0]:
            field = field.replace_data(np.clip(field.matrix, clim[0], None))

    if force_symmetric_clim:
        c = np.max(abs(field.matrix))
        kwargs['clim'] = (-c, c)
        if 'cmap' not in kwargs:
            kwargs['cmap'] = pp.plotting.plotter_matplotlib.MatplotlibPlotter.symmap

    field = field.squeeze()

    color_image = field.dimensions == 3
    if color_image:
        field = field/np.max(field.matrix)

    if all(field.islinear()):
        if 'origin' not in kwargs:
            kwargs['origin'] = 'lower'
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'

        im = ax.imshow(np.moveaxis(field.matrix, 0, 1), extent = field.extent[:4], **kwargs)
    elif not color_image:
        x, y = [ax.grid_node for ax in field.axes]
        im = ax.pcolormesh(x, y, np.moveaxis(field.matrix, 0, 1), **kwargs)
    else:
        raise ValueError("color images with non-linear axes not supported by this function.")

    ax.set_xlabel('{} [{}]'.format(field.axes[0].name, field.axes[0].unit))
    ax.set_ylabel('{} [{}]'.format(field.axes[1].name, field.axes[1].unit))
    if not color_image:
        colorbar = ax.get_figure().colorbar(im, ax = ax)
        colorbar.set_label('{} [{}]'.format(field.name, field.unit))
    else:
        ax.set_title(field.name)
    return im


def plot_field_1d(ax, field, *args, **kwargs):
    """
    Plots a 1d `field` as a line on `ax`.
    Uses as.plot() by default, this may be changed using keyword argument `plot_method`
    which takes the plot method's name, e.g. `plot_method='semilogy'` will create a semilog
    plot.
    All other arguments and keyword arguments are passed through to the plot method.
    """
    plot_method = kwargs.pop('plot_method', 'plot')
    field = field.squeeze()
    return getattr(ax, plot_method)(field.grid, field, *args, **kwargs)


def plot_fields_1d(ax, *fields, **kwargs):
    """
    Used to plot multiple 1d `fields` as lines on `ax`.

    Keyword arguments used by this function:
    `common_name`: A common name to be written on the y Axis label
    `plot_args`: a dictionary mapping from the index `i` of a field in `fields` to a `list`
                 of additional positional arguments to apply when plotting the `i`th field
    `plot_kwargs`: a dictionary mapping from the index `i` of a field in `fields` to a `dict`
                   of additional keyword arguments to apply when plotting the `i`th field
    All remaining keyword arguments are passed to each plot but may be overriden by arguments
    given in `plot_kwargs`.
    """
    common_name = kwargs.pop('common_name', None)
    plot_args = kwargs.pop('plot_args', dict())
    plot_kwargs = kwargs.pop('plot_kwargs', dict())

    for i, field in enumerate(fields):
        args = plot_args.get(i, tuple())
        field_kwargs = dict()
        field_kwargs.update(kwargs)
        field_kwargs.update(plot_kwargs.get(i, dict()))
        if 'label' not in field_kwargs:
            field_kwargs['label'] = field.name

        plot_field_1d(ax, field, *args, **field_kwargs)

    xAxis = field.axes[0]
    ax.set_xlabel('{} [{}]'.format(xAxis.name, xAxis.unit))

    yName = common_name if common_name else field.name
    ax.set_ylabel('{} [{}]'.format(yName, field.unit))
