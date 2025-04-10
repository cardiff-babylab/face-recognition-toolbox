# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Define all the YOLO model files - each model that might be used
model_files = [
    ('data/yolov8n-face.pt', './data/yolov8n-face.pt', 'DATA'),
    ('data/yolov8m-face.pt', './data/yolov8m-face.pt', 'DATA'),
    ('data/yolov8l-face.pt', './data/yolov8l-face.pt', 'DATA'),
    ('data/yolov11m-face.pt', './data/yolov11m-face.pt', 'DATA'),
    ('data/yolov11l-face.pt', './data/yolov11l-face.pt', 'DATA'),
]

# Icon files
icon_files = [
    ('graphics/icon_Z71_icon.ico', './graphics/icon_Z71_icon.ico', 'DATA'),
    ('graphics/icon_Z71_icon.png', './graphics/icon_Z71_icon.png', 'DATA'),
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data/yolov8n-face.pt', 'data'),
        ('data/yolov8m-face.pt', 'data'),
        ('data/yolov8l-face.pt', 'data'),
        ('data/yolov11m-face.pt', 'data'),
        ('data/yolov11l-face.pt', 'data'),
        ('graphics/icon_Z71_icon.ico', 'graphics'),
        ('graphics/icon_Z71_icon.png', 'graphics'),
    ],
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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 
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
)
