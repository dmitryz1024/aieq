from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .config import load_env_file
from .ui import MainWindow


def main() -> int:
    load_env_file()
    app = QApplication(sys.argv)
    app.setApplicationName("AIEQ")
    app.setOrganizationName("AIEQ")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
