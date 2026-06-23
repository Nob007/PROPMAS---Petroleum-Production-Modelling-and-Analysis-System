# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for IPM Nodal Analysis Suite
# Run from project root:  pyinstaller IPM_Suite.spec

import os
block_cipher = None

a = Analysis(
    ['gui/app.py'],
    pathex=['.'],                    # project root must be on the path
    binaries=[],
    datas=[
        # Bundle the engine modules
        ('core/*.py', 'core'),
    ],
    hiddenimports=[
        'numpy', 'scipy', 'scipy.optimize', 'scipy.interpolate',
        'matplotlib', 'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_agg',
        'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
        'core.ipr', 'core.pvt', 'core.vlp', 'core.solver_other',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'PySide6', 'PyQt5', 'wx'],
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
    name='IPM_Suite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # ← no console window (--windowed)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='gui/icon.ico',  # ← uncomment and supply icon.ico to add an icon
)
