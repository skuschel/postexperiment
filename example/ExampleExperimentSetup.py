
import re
import datetime
import functools
import os.path as osp
import collections

import numexpr as ne

import numpy as np
import scipy.interpolate as spinterp

from postexperiment import *
from postexperiment.core import make_shotid

def SetupGeminiFileLoaders():
    generic_images = ['CsIElectron', 'CsIPositron', 'CrystalSpec1', 'Espec1', 'PrettyPic',
                      'FlexEspec', 'GammaProfile', 'GammaSpectrum', 'HAPG', 'HOPG',
                      'PinholeCam', 'RRSpec', 'SideSpec']
    for key in generic_images:
        Shot.diagnostics[key] = GetItem(key)

    Shot.diagnostics['GasPressure'] = GetItem('GasPressure')


def SetupPinholeCamDiagnostic():
    Shot.diagnostics['PinholeCamRaw'] = GetItem('PinholeCam')

    Shot.diagnostics['PinholeCam'] = Chain(Shot.diagnostics['PinholeCamRaw'],
                                           RemoveDeadAndHotPixels,
                                           RemoveLinearBackground(50, 50, 50, 0))

    Shot.diagnostics['PinholeCamTotal'] = Chain(Shot.diagnostics['PinholeCam'],
                                                IntegrateAxis(1),
                                                IntegrateAxis(0),
                                                GetAttr('matrix'))


def SetupCrystalSpec1Diagnostic():
    Shot.diagnostics['CrystalSpec1Raw'] = GetItem('CrystalSpec1')

    Shot.diagnostics['CrystalSpec1'] = Chain(Shot.diagnostics['CrystalSpec1Raw'],
                                             RemoveDeadAndHotPixels,
                                             RemoveLinearBackground(50, 50, 50, 0))

    Shot.diagnostics['CrystalSpec1Total'] = FilterLRU(Chain(Shot.diagnostics['CrystalSpec1'],
                                                      IntegrateAxis(1),
                                                      IntegrateAxis(0),
                                                      GetAttr('matrix')), maxsize=1024)


def SetupBurney1Diagnostic():
    Shot.diagnostics['Burney1Raw'] = GetItem('Burney1')

    Shot.diagnostics['Burney1'] = Chain(Shot.diagnostics['Burney1Raw'],
                                        RemoveDeadAndHotPixels)


class ParameterFinder(object):
    def __init__(self, *shot_id_fields):
        self.ShotId = make_shotid(*shot_id_fields)
        self.configurations = collections.OrderedDict()

    def add_configuration(self, shot_id_values, configuration_dict):
        self.configurations[self.ShotId.literal(*shot_id_values)] = configuration_dict
        self.configurations = collections.OrderedDict(sorted(self.configurations.items(), key=lambda kv: kv[0]))

    def __call__(self, shot, **kwargs):
        keys = list(self.configurations.keys())
        last_key = keys[0]
        shotkey = self.ShotId(shot)
        for key in keys:
            if key > shotkey:
                break

            last_key = key

        return self.configurations[last_key]


@FilterFactory
def ApplyProjectiveTransformGetConfiguration(field, parameter_finder, context=None, **kwargs):
    config = parameter_finder(context['shot'])
    transform = ApplyProjectiveTransform(config['transform_p'], config['axes_fine'])
    return transform(field, **kwargs)


@FilterFactory
def IntegrateCellsGetConfiguration(field, parameter_finder, context=None, **kwargs):
    config = parameter_finder(context['shot'])
    transform = IntegrateCells(config['axes_coarse'])
    return transform(field, **kwargs)


def SetupGammaSpectrumDiagnostic():
    GetConfiguration = ParameterFinder(('date', str))

    # Pixelpositionen und Indizes ausgezählt/ausgemessen anhand von
    # /home/expgemini2018/Data/20180323/20180323r002/20180323r002s019_GammaSpectrum.tif

    # in /home/expgemini2018/Calibration/GammaSpectrum/AbbildungGammaSpectrum.png
    # sind die gewählten Punkte eingefärbt:
    GammaSpectrum_points_ij = np.array([[ 1,  3],  # grün
                                        [ 1, 18],  # weiß
                                        [28,  1],  # rot
                                        [26, 23]]) # schwarz
    GammaSpectrum_points_xy = np.array([[925, 685],
                                        [926, 269],
                                        [172, 724],
                                        [246, 127]])
    conf = dict()
    conf['transform_p'] = calculate_projective_transform_parameters(
                                        GammaSpectrum_points_ij, GammaSpectrum_points_xy)
    conf['axes_fine'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,31.501,0.02)),
                         pp.Axis(name='j', grid_node=np.arange(-0.5,26.501,0.02))]
    conf['axes_coarse'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,31.6,1.)),
                           pp.Axis(name='j', grid_node=np.arange(-0.5,26.6,1.))]

    GetConfiguration.add_configuration(('20180323',), conf)

    # Pixelpositionen und Indizes ausgezählt/ausgemessen anhand von 20180328 r002 s028
    GammaSpectrum_points_ij = np.array([[ 3, -6],  # grün
                                        [ 3,  7],  # weiß
                                        [28, -6],  # rot
                                        [28,  7]]) # schwarz
    GammaSpectrum_points_xy = np.array([[934.5, 573.5],
                                        [935.5, 291. ],
                                        [382. , 560. ],
                                        [389.5, 280.5]])
    conf = dict()
    conf['transform_p'] = calculate_projective_transform_parameters(
                                        GammaSpectrum_points_ij, GammaSpectrum_points_xy)
    conf['axes_fine'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,44.51,0.02)),
                         pp.Axis(name='j', grid_node=np.arange(-12.5,15.51,0.02))]
    conf['axes_coarse'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,44.51,1.)),
                           pp.Axis(name='j', grid_node=np.arange(-12.5,15.51,1.))]

    GetConfiguration.add_configuration(('20180328',), conf)

    # Pixelpositionen und Indizes ausgezählt/ausgemessen anhand von 20180404 r003 s102
    GammaSpectrum_points_ij = np.array([[ 4, -6],  # grün
                                        [ 4,  7],  # weiß
                                        [31, -6],  # rot
                                        [31,  7]]) # schwarz
    GammaSpectrum_points_xy = np.array([[922, 352],
                                        [922, 643],
                                        [319, 340],
                                        [310, 627]])
    conf = dict()
    conf['transform_p'] = calculate_projective_transform_parameters(
                                        GammaSpectrum_points_ij, GammaSpectrum_points_xy)
    conf['axes_fine'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,44.51,0.02)),
                         pp.Axis(name='j', grid_node=np.arange(-14.5,14.51,0.02))]
    conf['axes_coarse'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,44.51,1.)),
                           pp.Axis(name='j', grid_node=np.arange(-14.5,14.51,1.))]

    GetConfiguration.add_configuration(('20180404',), conf)

    # Pixelpositionen und Indizes ausgezählt/ausgemessen anhand von 20180405 r003 s005
    GammaSpectrum_points_ij = np.array([[ 1, -7],  # grün
                                        [ 1,  7],  # weiß
                                        [31, -7],  # rot
                                        [31,  7]]) # schwarz
    GammaSpectrum_points_xy = np.array([[977, 335],
                                        [978, 641],
                                        [323, 321],
                                        [314, 625]])
    conf = dict()
    conf['transform_p'] = calculate_projective_transform_parameters(
                                        GammaSpectrum_points_ij, GammaSpectrum_points_xy)
    conf['axes_fine'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,44.51,0.02)),
                         pp.Axis(name='j', grid_node=np.arange(-14.5,14.51,0.02))]
    conf['axes_coarse'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,44.51,1.)),
                           pp.Axis(name='j', grid_node=np.arange(-14.5,14.51,1.))]

    GetConfiguration.add_configuration(('20180405',), conf)

    Shot.diagnostics['GammaSpectrumConfiguration'] = GetConfiguration

    Shot.diagnostics['GammaSpectrumRaw'] = GetItem('GammaSpectrum')

    Shot.diagnostics['GammaSpectrum'] = Chain(Shot.diagnostics['GammaSpectrumRaw'],
                                              RemoveDeadAndHotPixels,
                                              RemoveLinearBackground(50, 50, 50, 50))

    Shot.diagnostics['GammaSpectrumProjected'] = Chain(Shot.diagnostics['GammaSpectrum'],
                                                       ApplyProjectiveTransformGetConfiguration(GetConfiguration),
                                                       SetFieldNameUnit(name='counts per cell', unit=''))

    Shot.diagnostics['GammaSpectrumCellwise'] = FilterLRU(Chain(Shot.diagnostics['GammaSpectrumProjected'],
                                                                IntegrateCellsGetConfiguration(GetConfiguration)), maxsize=1024)

    Shot.diagnostics['GammaSpectrumTotal'] = FilterLRU(Chain(Shot.diagnostics['GammaSpectrumCellwise'],
                                                       IntegrateAxis(1),
                                                       IntegrateAxis(0),
                                                       GetAttr('matrix')), maxsize=1024)

    Shot.diagnostics['GammaSpectrumDepthProfile'] = Chain(Shot.diagnostics['GammaSpectrumCellwise'],
                                                          IntegrateAxis(1),
                                                          SetFieldNameUnit(name='counts per row', unit=''))

    Shot.diagnostics['GammaSpectrumDepthProfileFit'] = Chain(Shot.diagnostics['GammaSpectrumDepthProfile'],
                                                             DoFit(polyexponential_1d))

    Shot.diagnostics['GammaSpectrumDepthProfileFitCurve'] = EvaluateFitResult(Shot.diagnostics['GammaSpectrumDepthProfile'],
                                                                              Shot.diagnostics['GammaSpectrumDepthProfileFit'],
                                                                              polyexponential_1d)

    Shot.diagnostics['GammaSpectrumDepthProfileGaussian'] = Chain(Shot.diagnostics['GammaSpectrumDepthProfile'],
                                                                  DoFit(gaussian_1d))

    Shot.diagnostics['GammaSpectrumTransverseProfile'] = Chain(Shot.diagnostics['GammaSpectrumCellwise'],
                                                               IntegrateAxis(0),
                                                               SetFieldNameUnit(name='counts per stack', unit=''))

    Shot.diagnostics['GammaSpectrumTransverseProfileGaussian'] = Chain(Shot.diagnostics['GammaSpectrumTransverseProfile'],
                                                                       DoFit(gaussian_1d))

    Shot.diagnostics['GammaSpectrumTransversePosition'] = Chain(Shot.diagnostics['GammaSpectrumTransverseProfileGaussian'],
                                                                GetAttr('center'))


def SetupCsIPositronDiagnostic():
    GetConfiguration = ParameterFinder(('date', str))

    # Pixelpositionen und Indizes ausgezählt/ausgemessen anhand von
    # /home/expgemini2018/Data/20180321/20180321r006/20180321r006s022_CsIPositron.tif

    # in /home/expgemini2018/Calibration/CsIPositron/AbbildungCsIPositron.png
    # sind die gewählten Punkte eingefärbt:
    CsIPositron_points_ij = np.array([[ 3,  3],  # grün
                                      [ 3, 11],  # weiß
                                      [16,  3],  # rot
                                      [16, 11]]) # schwarz
    CsIPositron_points_xy = np.array([[ 346,  314],
                                      [ 356,  750],
                                      [1047,  299],
                                      [1056,  737]])

    conf = dict()
    conf['transform_p'] = calculate_projective_transform_parameters(
                                            CsIPositron_points_ij, CsIPositron_points_xy)
    conf['axes_fine'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,19.501,0.01)),
                         pp.Axis(name='j', grid_node=np.arange(-0.5,14.501,0.01))]
    conf['axes_coarse'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,19.51,1.)),
                           pp.Axis(name='j', grid_node=np.arange(-0.5,14.51,1.))]

    GetConfiguration.add_configuration(('20180321',), conf)

    # Pixelpositionen und Indizes ausgezählt/ausgemessen anhand von
    # /home/expgemini2018/Data/20180404/20180404r003/20180404r003s045_CsIPositron.tif
    # gelten aber auch schon für 20180328!

    # in /home/expgemini2018/Calibration/CsIPositron/AbbildungCsIPositron2.png
    # sind die gewählten Punkte eingefärbt:
    CsIPositron_points_ij = np.array([[ 3,  4],  # grün
                                      [ 2, 12],  # weiß
                                      [16,  6],  # rot
                                      [16, 11]]) # schwarz
    CsIPositron_points_xy = np.array([[ 406,  362],
                                      [ 354,  759],
                                      [1048,  460],
                                      [1046,  710]])

    conf = dict()
    conf['transform_p'] = calculate_projective_transform_parameters(
                                            CsIPositron_points_ij, CsIPositron_points_xy)

    conf['axes_fine'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,19.501,0.01)),
                         pp.Axis(name='j', grid_node=np.arange(-0.5,14.501,0.01))]

    conf['axes_coarse'] = [pp.Axis(name='i', grid_node=np.arange(-0.5,19.51,1.)),
                           pp.Axis(name='j', grid_node=np.arange(-0.5,14.51,1.))]

    GetConfiguration.add_configuration(('20180328',), conf)

    Shot.diagnostics['CsIPositronConfiguration'] = GetConfiguration

    def ConvertCountsToMeV(field, context=None, **kwargs):
        shot = context['shot']
        if 'csip_gain' not in shot:
            return field

        try:
            gain = float(shot['csip_gain'])
        except ValueError:
            gain = None

        if gain is None:
            return field

        # Der Gain ist nicht linear. Im Handbuch steht etwas von 150V sind ein Faktor 10.
        # Das kommt auch ungefähr mit dem hin, was wir hier gemessen haben.

        # gain ausgemessen anhand von
        # /home/expgemini2018/Calibration/CsIPositron/Na22_reference_1600VGain_65600MeVdepo_161209_0089.tif
        # siehe
        # /home/expgemini2018/Calibration/CsIPositron/CsIPositron-20180404-Gain.html
        field = field * 10**((1600.-gain)/150.) / 30100.5
        field.name = field.name.replace('counts', 'MeV')
        field.unit = 'MeV'

        return field

    Shot.diagnostics['CsIPositronRaw'] = GetItem('CsIPositron')

    Shot.diagnostics['CsIPositron'] = Chain(Shot.diagnostics['CsIPositronRaw'],
                                            SetFieldNameUnit(name='counts per pixel', unit=''),
                                            RemoveLinearBackground(50, 50, 50, 50))

    Shot.diagnostics['CsIPositronMeV'] = Chain(Shot.diagnostics['CsIPositron'],
                                                       ConvertCountsToMeV)

    Shot.diagnostics['CsIPositronProjected'] = Chain(Shot.diagnostics['CsIPositron'],
                                                     ApplyProjectiveTransformGetConfiguration(GetConfiguration),
                                                     SetFieldNameUnit(name='counts per cell', unit=''))

    Shot.diagnostics['CsIPositronProjectedMeV'] = Chain(Shot.diagnostics['CsIPositronProjected'],
                                                       ConvertCountsToMeV)

    Shot.diagnostics['CsIPositronCellwise'] = FilterLRU(Chain(Shot.diagnostics['CsIPositronProjected'],
                                                              IntegrateCellsGetConfiguration(GetConfiguration)), maxsize=1024)

    Shot.diagnostics['CsIPositronCellwiseMeV'] = Chain(Shot.diagnostics['CsIPositronCellwise'],
                                                       ConvertCountsToMeV)

    Shot.diagnostics['CsIPositronGaussian'] = FilterLRU(Chain(Shot.diagnostics['CsIPositronCellwise'],
                                                               DoFit(gaussian_2d)), maxsize=1024)

    Shot.diagnostics['CsIPositronHorizontal'] = Chain(Shot.diagnostics['CsIPositronCellwise'],
                                                       IntegrateAxis(1),
                                                       SetFieldNameUnit(name='counts per column', unit=''))

    Shot.diagnostics['CsIPositronHorizontalMeV'] = Chain(Shot.diagnostics['CsIPositronHorizontal'],
                                                         ConvertCountsToMeV)

    Shot.diagnostics['CsIPositronHorizontalGaussian'] = Chain(Shot.diagnostics['CsIPositronHorizontalMeV'],
                                                               DoFit(gaussian_1d))

    Shot.diagnostics['CsIPositronVertical'] = Chain(Shot.diagnostics['CsIPositronCellwise'],
                                                     IntegrateAxis(0),
                                                     SetFieldNameUnit(name='counts per row', unit=''))

    Shot.diagnostics['CsIPositronVerticalMeV'] = Chain(Shot.diagnostics['CsIPositronVertical'],
                                                         ConvertCountsToMeV)

    Shot.diagnostics['CsIPositronVerticalGaussian'] = Chain(Shot.diagnostics['CsIPositronVertical'],
                                                             DoFit(gaussian_1d))

    Shot.diagnostics['CsIPositronTotal'] = FilterLRU(Chain(Shot.diagnostics['CsIPositronCellwise'],
                                                     IntegrateAxis(1),
                                                     IntegrateAxis(0),
                                                     GetAttr('matrix')), maxsize=1024)


    Shot.diagnostics['CsIPositronTotalMeV'] = Chain(Shot.diagnostics['CsIPositronCellwiseMeV'],
                                                    IntegrateAxis(1),
                                                    IntegrateAxis(0),
                                                    GetAttr('matrix'))

    Shot.diagnostics['CsIPositronHPos'] = Chain(Shot.diagnostics['CsIPositronHorizontalGaussian'],
                                                 GetAttr('center'))

    Shot.diagnostics['CsIPositronVPos'] = Chain(Shot.diagnostics['CsIPositronVerticalGaussian'],
                                                 GetAttr('center'))

    def CsIPositronYield(field, min_nrg=200, **kwargs):
        counts = field.matrix.reshape(-1)
        #total_yield = np.where(counts > min_nrg, counts, 0).sum()
        total_yield = ne.evaluate('sum(where(counts > min_nrg, counts, 0))')
        #pix_cnt = np.where(counts > min_nrg, 1, 0).sum()
        pix_cnt = ne.evaluate('sum(where(counts > min_nrg, 1, 0))')
        return pix_cnt, total_yield

    Shot.diagnostics['CsIPositronYield'] = FilterLRU(Chain(Shot.diagnostics['CsIPositronCellwiseMeV'],
                                                        CsIPositronYield), maxsize=1024)



def SetupEspec1Diagnostic(y_roi=(4., 2556.)):
    calib_dir = "/home/expgemini2018/Calibration/Espec/"
    calib_nrg = osp.join(calib_dir, 'px_to_MeV_rough_20180321.txt')
    nrg_calib = np.loadtxt(calib_nrg)
    n = len(nrg_calib)
    nrg_fun = spinterp.interp1d(np.linspace(0, n-1, n), nrg_calib, fill_value="extrapolate", bounds_error=False)

    slices = [slice(None), slice(None)]
    if y_roi is not None:
        slices[1] = slice(*y_roi)

    Shot.diagnostics['Espec1Raw'] = Chain(GetItem('Espec1'),
                                          Rotate180(),
                                          SliceField(slices))

    Shot.diagnostics['Espec1'] = Chain(GetItem('Espec1'),
                                       Rotate180(),
                                       RemoveDeadAndHotPixels,
                                       RemoveLinearBackground(100, 100, 100, 100),
                                       SliceField(slices))

    def ConvertEspecCountsToCharge(field, **kwargs):
        field_pc = field * 8e-6
        return field_pc

    Shot.diagnostics['Espec1Mapped'] = FilterLRU(Chain(Shot.diagnostics['Espec1'],
                                                 MapAxisGrid(0, nrg_fun),
                                                 SetFieldNameUnit(name='counts per MeV and transverse pixel', unit='1/MeV'),
                                                 SetAxisNameUnit(0, name='E', unit='MeV')), maxsize=200)

    Shot.diagnostics['Espec1MappedCharge'] = FilterLRU(Chain(Shot.diagnostics['Espec1Mapped'],
                                                             ConvertEspecCountsToCharge,
                                                             SetFieldNameUnit(name='charge per MeV and transverse pixel', unit='pC/MeV')), maxsize=200)

    Shot.diagnostics['Espec1MappedLinearAxis'] = FilterLRU(Chain(Shot.diagnostics['Espec1Mapped'],
                                                                 MakeAxesLinear()), maxsize=200)

    Shot.diagnostics['Espec1MappedChargeLinearAxis'] = FilterLRU(Chain(Shot.diagnostics['Espec1MappedCharge'],
                                                                       MakeAxesLinear()), maxsize=200)

    Shot.diagnostics['Espec1Spectrum'] = FilterLRU(Chain(Shot.diagnostics['Espec1Mapped'],
                                                         IntegrateAxis(1),
                                                         SetFieldNameUnit(name='counts per MeV', unit='1/MeV')), maxsize=1024)

    Shot.diagnostics['Espec1SpectrumCharge'] = FilterLRU(Chain(Shot.diagnostics['Espec1MappedCharge'],
                                                         IntegrateAxis(1),
                                                         SetFieldNameUnit(name='charge per MeV', unit='pC/MeV')), maxsize=1024)



def SetupGeminiDefaultDiagnostics():
    SetupGeminiFileLoaders()
    SetupGammaSpectrumDiagnostic()
    SetupCsIPositronDiagnostic()
    SetupPinholeCamDiagnostic()
    SetupCrystalSpec1Diagnostic()
    SetupBurney1Diagnostic()


class GeminiLogFileSource(object):
    def __init__(self, logfile, invalid_gsn=[0], date=None, run=None):
        self.logfile = logfile

        # gsn numbers where log.txt should not be trusted
        self.invalid_gsn = invalid_gsn
        self.date = date
        self.run = run

    def __call__(self):
        shots = collections.OrderedDict()
        shotid_re = re.compile('(\d*)r(\d*)s(\d*)')

        gsn_seen = set(self.invalid_gsn)

        for line in open(self.logfile):
            dtstring, shotid, gsn = line.strip().split("\t")

            date, time = dtstring.split()
            year, month, day = date.split('/')
            hour, minute, second = time.split(':')
            second, fractional = divmod(float(second), 1.0)
            second = int(round(second))
            us = int(round(fractional*1e6))
            if second == 60:
                dt = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), second-1, us, tzinfo=datetime.timezone.utc)
                dt += datetime.timedelta(0, 1)
            else:
                dt = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), second, us, tzinfo=datetime.timezone.utc)

            shotid = shotid_re.match(shotid)
            date, run, shotnum = shotid.group(1), int(shotid.group(2)), int(shotid.group(3))

            gsn = int(gsn)

            if self.date and date != self.date:
                continue

            if self.run and run != self.run:
                continue

            shot = Shot()
            shot['date'] = date
            shot['run'] = run
            shot['shot'] = shotnum

            if gsn in gsn_seen:
                shot['gsn'] = 'NA'
            else:
                shot['gsn'] = gsn
            gsn_seen.add(gsn)

            shot['datetime'] = dt
            shot['timestamp'] = dt.timestamp()


            shots[(date, run, shotnum)] = shot

        return list(shots.values())


class RawOrImageLoader():
    def __init__(self, RawReader):
        self.RawReader = RawReader

    def __call__(self, fname):
        if fname.lower().endswith('raw'):
            return self.RawReader(fname)
        return ImageLoader(fname)

def LoadGasPressure(filename):
    data = np.fromfile(filename)
    tAx = pp.Axis(name='t', unit='s', grid=data[:10000])
    fields = [pp.Field(data[(i+1)*10000:(i+2)*10000],
                    name='Channel {}'.format(i+1),
                    unit='V',
                    axes=[tAx])
            for i in range(4)
            ]
    return fields


def GetGeminiShotSeries(date=None, run=None, logfile_invalid_gsn = [0, 197588]):
    data_root = '/home/expgemini2018/Data/'
    files_root = data_root

    if date is not None:
        files_root = osp.join(files_root, date)
        if run is not None:
            files_root = osp.join(files_root, "{}r{:03d}".format(date, run))

    FileReaders = dict()
    FileReaders['Burney1'] = Make_LazyReader(RawOrImageLoader(RawReader('Burney1', 640, 480, dtype='>u2')))
    FileReaders['Burney2'] = Make_LazyReader(RawOrImageLoader(RawReader('Burney2', 640, 480, dtype='>u2')))
    FileReaders['Eprofile'] = Make_LazyReader(RawOrImageLoader(RawReader('Eprofile', 640, 480, dtype='>u2')))
    FileReaders['CeYAGscreen'] = Make_LazyReader(RawOrImageLoader(RawReader('CeYAGscreen', 640, 480, dtype='>u2')))
    FileReaders['GasPressure'] = Make_LazyReader(LoadGasPressure)

    shots = ShotSeries(('date', str), ('run', int), ('shot', int))
    shots.sources['filesource'] = FileSource(files_root,
                                             r'(2018\d{4})r(\d{3})s(\d{3})_(.*)\..*', 4,
                                             {1: ('date', None), 2: ('run', int), 3: ('shot', int)},
                                             FileReaders = FileReaders)

    shots.sources['logfile'] = GeminiLogFileSource(osp.join(data_root, 'log.txt'), invalid_gsn = logfile_invalid_gsn, date=date, run=run)

    return shots.load()
