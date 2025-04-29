# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# Define data directory models
data_dir = os.path.join(os.path.abspath('.'), 'data')
graphics_dir = os.path.join(os.path.abspath('.'), 'graphics')

# Model files to include
model_files = [
    os.path.join(data_dir, 'yolov8n-face.pt'),
    os.path.join(data_dir, 'yolov8m-face.pt'),
    os.path.join(data_dir, 'yolov8l-face.pt'),
    os.path.join(data_dir, 'yolov11m-face.pt'),
    os.path.join(data_dir, 'yolov11l-face.pt'),
]

# Graphics files to include
graphics_files = [
    os.path.join(graphics_dir, 'icon_Z71_icon.ico'),
    os.path.join(graphics_dir, 'icon_Z71_icon.png'),
    os.path.join(graphics_dir, 'toolbox_splash.png'),
]

# Collect all data files
datas = []

# Add model files if they exist
for model_file in model_files:
    if os.path.exists(model_file):
        datas.append((model_file, 'data'))
    else:
        print(f"Warning: Model file {model_file} not found")

# Add graphics files if they exist
for graphics_file in graphics_files:
    if os.path.exists(graphics_file):
        datas.append((graphics_file, 'graphics'))
    else:
        print(f"Warning: Graphics file {graphics_file} not found")

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'ultralytics', 
        'ultralytics.nn.tasks',
        'ultralytics.yolo.engine',
        'ultralytics.yolo.utils',
        'cv2',
        'pandas',
        'numpy',
        'torch',
        'torch.nn',
        'retinaface',
        'PIL',
        'PIL._tkinter_finder',
        'requests',
        'tqdm',
        'pytz',
        'matplotlib',
        'matplotlib.pyplot',
        'pyi_splash',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'notebook', 
        'scipy.optimize', 
        'scipy.signal', 
        'scipy.sparse', 
        'scipy.special',
        'IPython',
        'ipykernel',
        'jedi',
        'jinja2', 
        'nbconvert',
        'nbformat',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FaceDetectionApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['graphics/icon_Z71_icon.ico'],
    version='file_version_info.txt',
    splash='graphics/toolbox_splash.png',
)
