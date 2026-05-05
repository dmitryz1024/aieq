from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .config import load_env_file
from .ui import MainWindow


def resource_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    candidates = [
        Path(getattr(sys, "_MEIPASS")) / relative if hasattr(sys, "_MEIPASS") else None,
        Path.cwd() / relative,
        Path(__file__).resolve().parent.parent / relative,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return Path(__file__).resolve().parent.parent / relative


def configure_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AIEQ.AIEQ.Prototype")
    except Exception:
        pass


def main() -> int:
    load_env_file()
    configure_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("AIEQ")
    app.setOrganizationName("AIEQ")

    icon_path = resource_path("assets/icon.ico")
    if not icon_path.exists():
        icon_path = resource_path("assets/icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
