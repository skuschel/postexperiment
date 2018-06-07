'''
helper functions providing the LabBook datasource to retrieve data from a
human written labbook in form of a Spreadsheet, e.g. google docs.

Copyright:
Stephan Kuschel, 2018
'''


import numpy as np
import copy

__all__ = ['LabBookSource']


class LabBookSource():
    '''
    Creates a list of `Shot`s from given csv data downloadable from google docs.

    Stephan Kuschel, 2018
    '''
    def __init__(self, link, continued_int_id_field, **kwargs):
        self.link = link
        self.continued_int_id_field = continued_int_id_field
        self.kwargs = kwargs

    def __call__(self):
        full_shotlist = create_full_shotlist_from_googledocs(self.link,
                                        self.continued_int_id_field, **self.kwargs)
        return full_shotlist


def create_full_shotlist_from_googledocs(link, continued_int_id_field,
                                        header=1, rowstart=2, rowend=None,
                                        isvalidentryf=lambda h,d: d is not None and d != '',
                                        reset_discontinued=True,
                                        isvalidrowf=lambda entry: True):
    '''
    creates the full shotlist from a google docs link, which downloads the
    shotshet as csv.
    In google docs use: File -> Download as -> comma separated vales (current sheet)
    and use this downloadlink here.
    '''

    # download shotlog from google docs
    import requests
    r = requests.get(link)
    # csv
    import csv
    tabledata = list(csv.reader(iter(r.content.decode().splitlines())))
    # list of one dict per row
    shotlog_entries = create_shotlog_entry_list(tabledata, header=header, rowstart=rowstart,
                                                rowend=rowend, isvalidentryf=isvalidentryf)
    # list of one dict per shot
    full_shotlist = create_full_shotlist(shotlog_entries, continued_int_id_field,
                                        reset_discontinued=reset_discontinued,
                                        isvalidrowf=isvalidrowf)
    return full_shotlist


def create_shotlog_entry_list(tabledata, header=1, rowstart=2, rowend=None,
                              isvalidentryf=lambda h,d: d is not None and d != ''):
    '''
    creates a list of dictionaries containing the information of the shotlog (labbook) table,
    given by shotlogdata.

    args
    ----
    tabledata
      This date must be a
      list of list of strings (columns times rows with strings of entries).

    kwargs
    ------
    header=1
      index of the headerrow: This will be used for indexing of the data and must be
      a unique column identifier.
    rowstart=2
      first row containing data
    rowend=None
      last row containing data
    isvalidentryf=lambda h,d: d is not None and d != ''
      callable returning true or false to tell, if an enty should be ignored.
      The function will be called with the identifier of the header and the actual data.
      The default returns false if the data is either `None` or `''`.
    '''
    header = tabledata[header]
    ret = list()
    for row in tabledata[rowstart: rowend]:
        # only add valid entries, keep empy cells without dict entry
        ret.append({k:v for k,v in zip(header, row) if isvalidentryf(k, v)})
    return ret


def create_full_shotlist(shotlog_entries, continued_int_id_field,
                         reset_discontinued=True,
                         isvalidrowf=lambda entry: True):
    '''
    This function takes the ouput of `create_shotlog_entry_list` and returns
    a list containing one entry for EVERY shot. In the labbook people might
    only write down every 20th shot, whenever parameters change, then
    this function expands the list to 20 entries. The list of consecutive
    shots is identified by the second argument `continued_int_id_field`,
    which must be an `int` in the `shotlog_entries`.

    The function also infers the parameters of a shot, if the
    value was written down only for an older shot. This continuation of old
    data is interrupted if
      * the shotnumer decreases, or
      * an invalid row is encountered, which is any row,
        where the command `int(shot[continued_int_id_field])` fails, or
      * the function `isvalidrowf(shotlog_entry)` returns false.
        By default this function never returns false
        and is defined as `isvalidrowf=lambda entry: True`.

    Stephan Kuschel, 2018
    '''
    ret = list()
    rowdict = dict()
    sn_last = None
    for shotlog_entry in shotlog_entries:
        if not isvalidrowf(shotlog_entry):
            rowdict = dict()  # do not continue old data
        try:
            sn = int(shotlog_entry[continued_int_id_field])
        except(ValueError, KeyError):
            sn = None

        if sn_last is None:
            if reset_discontinued:
                rowdict = dict()
            if sn is not None:
                # add this single shot
                rowdict.update(shotlog_entry)
                ret.append(copy.copy(rowdict))
                sn_last = sn
        elif sn is None:
            sn_last = None  # reset on next loop
        else:
            assert sn is not None
            assert sn_last is not None
            if sn < sn_last:
                sn_last = None
                continue
            # fill between sn_last and sn-1
            for n in range(sn_last + 1, sn):
                rowdict[continued_int_id_field] = n
                ret.append(copy.copy(rowdict))
            rowdict.update(shotlog_entry)
            ret.append(copy.copy(rowdict))
            sn_last = sn

    return ret
