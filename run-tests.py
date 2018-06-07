#!/usr/bin/env python


# THIS FILE MUST RUN WITHOUT ERROR ON EVERY COMMIT!


import sys
import os


def runcmd(cmd):
    '''
    run command cmd and exit if it fails.
    '''
    import subprocess
    print('=====  running next command =====')
    print('$ ' + cmd)
    exitstatus = subprocess.call(cmd, shell=True)
    if exitstatus == 0:
        print('OK')
    else:
        print('run-tests.py failed. aborting.')
        print('The failing command was:')
        print(cmd)
        exit(exitstatus)


def run_autopep8(args):
    import autopep8
    if autopep8.__version__ < '1.2':
        print('upgrade to autopep8 >= 1.2 (installed: {:})'.format(
            str(autopep8.__version__)))
        exit(1)
    autopep8mode = '--in-place' if args.autopep8 == 'fix' else '--diff'
    argv = ['autopep8', '-r', 'postpic', '--ignore-local-config', autopep8mode,
            '--ignore=W391,E123,E226,E24', '--max-line-length=99']
    print('===== running autopep8 =====')
    print('autopep8 version: ' + autopep8.__version__)
    print('$ ' + ' '.join(argv))
    autopep8.main(argv)


def run_alltests(python='python', fast=False, skip_setup=False):
    '''
    runs all tests on postpic. This function has to exit without error on every commit!
    '''
    python += ' '
    # make sure .pyx sources are up to date and compiled
    if not skip_setup:
        runcmd(python + 'setup.py develop --user')

    # find pep8 or pycodestyle (its successor)
    pycodestylecmd = None
    try:
        import pep8
        pycodestylecmd = 'pep8'
    except(ImportError):
        pass
    try:
        import pycodestyle
        pycodestylecmd = 'pycodestyle'
    except(ImportError):
        pass
    if not pycodestylecmd:
        raise ImportError('Install pep8 or pycodestyle (its successor)')

    cmds = [python + '-m nose --exe',
            python + '-m ' + pycodestylecmd +
            ' postexperiment --statistics --count --show-source '
            '--ignore=W391,E123,E226,E24,W504 --max-line-length=99']
    cmdo = []
    if not fast:
        cmds += cmdo
    for cmd in cmds:
        runcmd(cmd)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='''
        Without arguments this runs all tests
        on the postpic codebase.
        This script MUST EXIT WITHOUT ERROR on EVERY commit!
        ''')
    parser.add_argument('--autopep8', default='', nargs='?', metavar='fix',
                        help='''
        run "autopep8" on the codebase.
        Use "--autopep8" to preview changes. To apply them, use
        "--autopep8 fix"
        ''')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Only run a subset of tests. Used for commit hook.')
    parser.add_argument('--skip-setup', action='store_true', default=False,
                        help='Skip the "setup.py develop --user" step.')
    pyversiongroup = parser.add_mutually_exclusive_group(required=False)
    pyversiongroup.add_argument('--pycmd', default='python{}'.format(sys.version_info.major),
                                help='use "PYCMD" as python interpreter for all subcommands. '
                                'default: "pythonX" where X is sys.version_info.major. '
                                'This means that by default the same version is used that '
                                'executes this script.'
                                )
    pyversiongroup.add_argument('-2', action='store_const', dest='pycmd', const='python2',
                                help='same as "--pycmd python2"')
    pyversiongroup.add_argument('-3', action='store_const', dest='pycmd', const='python3',
                                help='same as "--pycmd python3"')
    args = parser.parse_args()

    if args.autopep8 != '':
        run_autopep8(args)
        exit()

    run_alltests(args.pycmd, fast=args.fast, skip_setup=args.skip_setup)


if __name__ == '__main__':
    main()