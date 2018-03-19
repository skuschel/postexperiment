
import os
import os.path as osp
import re

from .shot import Shot, ShotSeries

def FileSeries(dirname, pattern, filekey, fields):
    """
    Produces a list of `Shot`s, given a filename `pattern`, a directory `dirname` and a description of `fields`
    that shall be extracted from the file names. The filepath will be stored under the key given by `filekey`.
    """
    pattern = re.compile(pattern)
    names = os.listdir(dirname)
    shots = ShotSeries()
    for name in names:
        path = osp.join(dirname, name)
        if not osp.isfile(path):
            continue
        match = pattern.match(name)
        if not match:
            continue

        shotinfo = dict()
        shotinfo['name'] = osp.splitext(name)[0]
        shotinfo[filekey] = path
        for i, (n, t) in fields.items():
            try:
                shotinfo[n] = t(match.group(i))
            except ValueError:
                shotinfo[n] = match.group(i)

        shots.append(Shot(**shotinfo))

    return shots
