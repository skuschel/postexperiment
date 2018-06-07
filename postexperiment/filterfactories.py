'''
Copyright:
Alexander Blinne, 2018
'''


import numpy as np
import scipy.ndimage

import postpic as pp

from . import common
from . import algorithms


@common.FilterFactory
def Chain(line, *args, context=None, **kwargs):
    """
    Applies a chain of filters
    """
    for f in args:
        line = f(line, context=context, **kwargs)
    return line


@common.FilterFactory
def FitInitialGuess(field, fitmodel, **kwargs):
    return fitmodel.initial_guess(field, **kwargs)


@common.FilterFactory
def DoFit(field, fitmodel, **kwargs):
    return fitmodel.do_fit(field, **kwargs)


@common.FilterFactory
def EvaluateFitResult(shot, fielddiagnostic, fitdiagnostic, fitfunction, **kwargs):
    field = fielddiagnostic(shot, **kwargs)
    p = fitdiagnostic(shot, **kwargs)
    return algorithms.field_evaluate(field, fitfunction(p))


@common.FilterFactory
def SubtractOffset(field, offset, **kwargs):
    return field - offset


@common.FilterFactory
def SumAxis(field, axis, summation_bounds=None, **kwargs):
    """
    Sums a field along one axis
    """
    if summation_bounds is not None:
        slices = [slice(None) for _ in range(field.dimensions)]
        slices[axis] = slice(*summation_bounds)
        field = field[slices]
    return field.sum(axis=axis)


@common.FilterFactory
def IntegrateAxis(field, axis, integration_bounds=None, **kwargs):
    """
    Sums a field along one axis
    """
    if integration_bounds is not None:
        slices = [slice(None) for _ in range(field.dimensions)]
        slices[axis] = slice(*integration_bounds)
        field = field[slices]
    return field.integrate(axis)


@common.FilterFactory
def LoadImage(shot, img_key, **kwargs):
    return pp.Field.importfrom(shot[img_key])


@common.FilterFactory
def GetAttr(obj, attrname, **kwargs):
    return getattr(obj, attrname)


@common.FilterFactory
def GetItem(obj, item, **kwargs):
    return obj[item]


@common.FilterFactory
def ApplyProjectiveTransform(field, transform_p, new_axes, **kwargs):
    def transform(i, j):
        return algorithms.projective_transform(transform_p, i, j)

    return field.map_coordinates(new_axes, transform)


@common.FilterFactory
def MapAxisGrid(field, axis, fun, context=None, map_axis_grid_kwargs=dict(), **kwargs):
    return field.map_axis_grid(axis, fun, **map_axis_grid_kwargs)


@common.FilterFactory
def MakeAxesLinear(field, *new_ax_lengths, context=None, **kwargs):
    axes = field.axes[:]
    for i, old_ax in enumerate(axes):
        if old_ax.islinear():
            continue
        old_grid = old_ax.grid
        n = len(old_grid)
        if i < len(new_ax_lengths) and new_ax_lengths[i] is not None:
            n = new_ax_lengths[i]
        new_grid = np.linspace(np.min(old_grid), np.max(old_grid), n)
        axes[i] = pp.Axis(name=old_ax.name, unit=old_ax.unit, grid=new_grid)

    return field.map_coordinates(axes, **kwargs)


@common.FilterFactory
def IntegrateCells(field, new_axes, **kwargs):
    shape = [len(ax) for ax in new_axes]
    field_integrated = pp.Field(
        np.zeros(shape), axes=new_axes, name=field.name, unit=field.unit)
    N, M = field_integrated.shape
    for i in range(N):
        for j in range(M):
            imin = field_integrated.axes[0].grid_node[i]
            imax = field_integrated.axes[0].grid_node[i + 1]
            jmin = field_integrated.axes[1].grid_node[j]
            jmax = field_integrated.axes[1].grid_node[j + 1]
            field_integrated.matrix[i, j] = field[imin:imax,
                                                  jmin:jmax].integrate().matrix

    return field_integrated


@common.FilterFactory
def SetFieldNameUnit(field, name=None, unit=None, **kwargs):
    if name:
        field.name = name
    if unit:
        field.unit = unit
    return field


@common.FilterFactory
def SetAxisNameUnit(field, axis, name=None, unit=None, **kwargs):
    if name:
        field.axes[axis].name = name
    if unit:
        field.axes[axis].unit = unit
    return field


@common.FilterFactory
def RemoveLinearBackground(field, border_left=100, border_right=100, border_bottom=100,
                           border_top=100, **kwargs):
    mask = np.zeros_like(field.matrix, dtype=bool)
    m, n = mask.shape
    a, b, c, d = border_left, border_right, border_bottom, border_top
    b = m - b
    d = m - d
    mask[a:b, c:d] = True
    return field.replace_data(algorithms.remove_linear_background_2d(field.matrix, mask))


@common.FilterFactory
def CropBorders(field, crop_left=0, crop_right=0, crop_bottom=0, crop_top=0, **kwargs):
    a, b, c, d = crop_left, crop_right, crop_bottom, crop_top
    b = -b if b > 0 else None
    d = -d if d > 0 else None
    return field[a:b, c:d]


@common.FilterFactory
def SliceField(field, slices, **kwargs):
    return field[slices]


@common.FilterFactory
def ClipValues(field, a, b, **kwargs):
    return field.replace_data(np.clip(field.matrix, a, b))


@common.FilterFactory
def Rotate90(field, k=1, axes=(0, 1), **kwargs):
    return field.rot90(k=k, axes=axes)


@common.FilterFactory
def Rotate180(field, axes=(0, 1), **kwargs):
    return field.rot90(k=2, axes=axes)


@common.FilterFactory
def Flip(field, axis, **kwargs):
    return field.flip(axis)


@common.FilterFactory
def GreyOpening(field, **kwargs):
    data = field.matrix

    kwargs2 = {k: v for k, v in kwargs.items() if k in [
        'structure', 'size', 'footprint', 'mode', 'cval', 'origin']}

    if data.ndim == 3 and data.shape[2] < 5:
        for i in range(data.shape[2]):
            data[..., i] = scipy.ndimage.morphology.grey_opening(
                data[..., i], **kwargs)
    else:
        data = scipy.ndimage.morphology.grey_opening(data, **kwargs)
    return field.replace_data(data)


@common.FilterFactory
def GreyClosing(field, **kwargs):
    data = field.matrix

    kwargs2 = {k: v for k, v in kwargs.items() if k in [
        'structure', 'size', 'footprint', 'mode', 'cval', 'origin']}

    if data.ndim == 3 and data.shape[2] < 5:
        for i in range(data.shape[2]):
            data[..., i] = scipy.ndimage.morphology.grey_closing(
                data[..., i], **kwargs)
    else:
        data = scipy.ndimage.morphology.grey_closing(data, **kwargs)
    return field.replace_data(data)


@common.FilterFactory
def Median(field, **kwargs):
    data = field.matrix

    kwargs2 = {k: v for k, v in kwargs.items() if k in [
        'size', 'footprint', 'mode', 'cval', 'origin']}

    if data.ndim == 3 and data.shape[2] < 5:
        for i in range(data.shape[2]):
            data[..., i] = scipy.ndimage.median_filter(data[..., i], **kwargs2)
    else:
        data = scipy.ndimage.median_filter(data, size=(3, 3))
    return field.replace_data(data)


@common.FilterFactory
def ReadRaw(shot, filekey, width, height, bands=1, bands_axis=2, dtype=np.uint16, **kwargs):
    fname = shot[filekey]
    d = np.fromfile(fname, dtype=dtype)

    shape = [height, width]
    if bands > 1:
        shape.insert(bands_axis, bands)

    d = d.reshape(shape)

    if bands > 1:
        # switch to pixel-inverleaved mode, default for matplotlib
        d = np.moveaxis(d, bands_index, 2)

    # switch to PostPic.Field compatible axes
    d = np.swapaxes(d, 0, 1)[:, ::-1, ...]

    # convert to float
    d = np.asfarray(d)

    axes = []
    axes.append(pp.Axis(name='x', unit='px',
                        grid=np.linspace(0, width - 1, width)))
    axes.append(pp.Axis(name='y', unit='px',
                        grid=np.linspace(0, height - 1, height)))

    if bands > 1:
        axes.append(pp.Axis(name='band', unit='',
                            grid=np.linspace(0, bands - 1, bands)))

    return pp.Field(d, name='filekey', unit='counts', axes=axes)


@common.FilterFactory
def MultiFormatLoader(shot, filekey, get_loader, **kwargs):
    loader = get_loader(shot, filekey)
    return loader(shot)
