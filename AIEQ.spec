# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


project_root = Path(SPECPATH)
assets_dir = project_root / "assets"
curves_dir = project_root / "curves"

datas = []
for asset in ("icon.png", "icon.ico"):
    path = assets_dir / asset
    if path.exists():
        datas.append((str(path), "assets"))

for subdir in ("devices", "targets"):
    folder = curves_dir / subdir
    if folder.exists():
        for path in folder.glob("*.txt"):
            datas.append((str(path), f"curves/{subdir}"))

icon_path = assets_dir / "icon.ico"

binaries = []
hiddenimports = [
    "numpy",
    "pyqtgraph",
    "sounddevice",
]
hiddenimports += collect_submodules("autoeq")
hiddenimports += collect_submodules("llama_cpp")
datas += collect_data_files("llama_cpp")
binaries += collect_dynamic_libs("llama_cpp")

a = Analysis(
    ["source/__main__.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIEQ",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AIEQ",
)
