# -*- mode: python ; coding: utf-8 -*-

import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include data files
        ('data/wsr_simulator/*', 'data'),
        ('assets/model_weights/*', 'model_weights'),
        ('assets/logs/*', 'logs'),
        ('utils/*', 'utils'),
        ('env/wsr', '.'),
    ],
    hiddenimports=[
        # Add any hidden imports here
        'torch', 'numpy', 'argparse', 'unittest.mock',
        'utils.arg_parser', 'utils.functions', 'utils.definitions',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter', 'test', 'pytest', 'matplotlib.tests', 'numpy.tests',
        'scipy.tests', 'pandas.tests', 'torch.test', 'PIL.tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Enable large file support
a.archivename = 'wsmart_route'  # This helps with large files

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # This can help with size issues
    name='wsmart_route',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Enable compression
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    asarchive=True,  # Important: Use archive mode for large applications
)