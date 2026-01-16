# -- mode: python ; coding: utf-8 --

# To build: pyinstaller simulator.spec

import os
import sys
import wandb
import osmnx
import hexaly

from glob import glob

# Define the path to the site-packages directory
wandb_path = os.path.dirname(wandb.__file__)
osmnx_path = os.path.dirname(osmnx.__file__)
hexaly_package_path = os.path.dirname(hexaly.__file__)

# Find the osmnx metadata directory
# This path is usually one level up from the osmnx package directory
try:
    osmnx_metadata_dir = glob(os.path.join(os.path.dirname(osmnx_path), 'osmnx-*.dist-info'))[0]
except IndexError:
    osmnx_metadata_dir = None

lib_source = os.path.join(hexaly_package_path, 'libhexaly140.so')

# Assuming the root of your project is where you want to start path searching
pathex=['.'] # Set pathex to the /app directory

a = Analysis(
    ['__main__.py'], # Explicitly list entry point
    pathex=['.'], # Start path search from here
    binaries=[
        (lib_source, '.'),
    ],
    datas=[
        # Include data files
        ('data/wsr_simulator/*', 'data'),
        ('assets/model_weights/*', 'assets/model_weights'),
        ('assets/logs/*', 'assets/logs'),

        # === Manually include specific utility files ===
        ('app/src/utils/arg_parser.py', 'app/src/utils'),
        ('app/src/utils/cryptography.py', 'app/src/utils'),
        ('app/src/utils/definitions.py', 'app/src/utils'),
        ('app/src/utils/functions.py', 'app/src/utils'),
        ('app/src/utils/graph_utils.py', 'app/src/utils'),
        ('app/src/utils/io_utils.py', 'app/src/utils'),
        ('app/src/utils/log_utils.py', 'app/src/utils'),
        ('app/src/utils/setup_utils.py', 'app/src/utils'),

        # Include the __init__.py file to make 'utils' a package
        ('app/src/utils/__init__.py', 'app/src/utils'),
    ],
    hiddenimports=[
        # Other modules
        'torch', 'numpy', 'argparse',
        'multiprocessing', 'torch.multiprocessing',
        'multiprocessing.pool', 'multiprocessing.manager',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter', 'test', 'pytest', 'matplotlib.tests', 'numpy.tests',
        'onnxscript', 'pysqlite2', 'MySQLdb', 'psycopg2', 'expecttest',
        'scipy.tests', 'pandas.tests', 'torch.test', 'PIL.tests', 'torchaudio',
        'hypothesis', 'torch.onnx._internal.fx.passes', 'importlib_resources.trees',
        'matplotlib.backends.backend_qt', 'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.qt_compat', 'PySide6', 'shiboken6',
    ],
    noarchive=False,
    optimize=0,
    cipher=None,
    key=None,
    # === Use collect_data for the required code directories ===
    collect_all=[
        'src.policies',
        'src.pipeline.simulator',
    ],
    collect_submodules=[],
    collect_data=[],
    collect_entrypoints=[],
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_clean_pkgname=[],
)

# Enable large file support
a.archivename = 'WSmartRoute'  # This helps with large files

# Add the Tree imports to a.datas
a.datas += Tree(wandb_path, prefix='wandb')
a.datas += Tree(osmnx_path, prefix='osmnx')
if osmnx_metadata_dir:
    a.datas += Tree(osmnx_metadata_dir, prefix=os.path.basename(osmnx_metadata_dir))

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='WSmartRouteSimulator', # Name of the executable file
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True, # Use 'False' for a GUI application (no command line window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='simulator' # Name of the final folder/bundle
)
