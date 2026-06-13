from __future__ import annotations

import sys

from PySide6.QtCore import QCoreApplication, QSettings


def main() -> int:
    app = QCoreApplication(sys.argv)
    app.setApplicationName("AIEQ")
    app.setOrganizationName("AIEQ")

    settings = QSettings()
    for key in ("window/geometry", "window/normal_geometry", "window/state", "window/main_splitter"):
        settings.remove(key)
    settings.sync()
    print(f"Window layout settings reset: {settings.fileName()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
