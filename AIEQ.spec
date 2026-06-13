# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


project_root = Path(SPECPATH)
assets_dir = project_root / "assets"
languages_dir = project_root / "languages"


def optional_collect_submodules(package):
    try:
        return collect_submodules(package)
    except Exception:
        return []


def optional_collect_data_files(package):
    try:
        return collect_data_files(package)
    except Exception:
        return []


def optional_collect_dynamic_libs(package):
    try:
        return collect_dynamic_libs(package)
    except Exception:
        return []


datas = []
if assets_dir.exists():
    for path in assets_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".ico", ".svg"}:
            datas.append((str(path), "assets"))

if languages_dir.exists():
    for path in languages_dir.glob("*.json"):
        datas.append((str(path), "languages"))

icon_path = assets_dir / "icon.ico"

binaries = []
hiddenimports = [
    "numpy",
    "pyqtgraph",
    "sounddevice",
]
hiddenimports += optional_collect_submodules("autoeq")
hiddenimports += optional_collect_submodules("llama_cpp")
datas += optional_collect_data_files("llama_cpp")
binaries += optional_collect_dynamic_libs("llama_cpp")

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
