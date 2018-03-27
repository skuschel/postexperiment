
import postpic as pp
import numpy as np

def field_imshow(field, ax, force_symmetric_clim=False, **kwargs):
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
        x, y = field.grid
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
    plot_method = kwargs.pop('plot_method', 'plot')
    field = field.squeeze()
    return getattr(ax, plot_method)(field.grid, field, *args, **kwargs)


def plot_fields_1d(ax, *fields, **kwargs):
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
