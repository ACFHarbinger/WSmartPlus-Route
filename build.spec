# -- mode: python ; coding: utf-8 --

# To build: pyinstaller app.spec

import PySide6
import os

pyside_path = os.path.dirname(PySide6.__file__)

# pathex=['.'] means it looks in the current directory (where app.spec is)

pathex=['/backend']

a = Analysis(
    ['__main__.py'],
    pathex=['/backend'],
    binaries=[],
    datas=[
        # Include data files
        ('data/wsr_simulator/*', 'data'),
        ('assets/model_weights/*', 'model_weights'),
        ('assets/logs/*', 'logs'),
    ],
    hiddenimports=[
        # Core modules PyInstaller often needs for PySide6/subprocess interactions
        'PySide6.QtSvg',
        'PySide6.QtXml',

        # Other modules
        'torch', 'numpy', 'argparse',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter', 'test', 'pytest', 'matplotlib.tests', 'numpy.tests', 
        'onnxscript', 'pysqlite2', 'MySQLdb', 'psycopg2', 'expecttest',
        'scipy.tests', 'pandas.tests', 'torch.test', 'PIL.tests', 'torchaudio',
        'hypothesis', 'torch.onnx._internal.fx.passes', 'importlib_resources.trees'
    ],
    noarchive=False,
    optimize=0,
    cipher=None,
    key=None,
    collect_all=[],
    collect_submodules=[],
    collect_data=[(pyside_path, 'PySide6')], # Include PySide6 data globally
    collect_entrypoints=[],
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_clean_pkgname=[],
)

# Enable large file support
a.archivename = 'WSmartRoute'  # This helps with large files

a.datas += Tree(os.path.join(pyside_path, 'Qt', 'plugins', 'platforms'), prefix='PySide6/Qt/plugins/platforms')

pyz = PYZ(a.pure, a.zipped_data,
    cipher=None
)


exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='WSmartRouteApp', # Name of the executable file
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # Use 'False' for a GUI application (no command line window)
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
    name='WSmartRoute' # Name of the final folder/bundle
)