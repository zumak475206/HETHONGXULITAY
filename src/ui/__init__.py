"""UI Module initialization.
Exports MainWindow, styling QSS, PiP widget, and background workers.
"""

from src.ui.theme import MODERN_DARK_QSS
from src.ui.main_window import MainWindow
from src.ui.pip_widget import PiPWidget
from src.ui.workers import CameraWorker, TrainWorker

__all__ = ["MainWindow", "MODERN_DARK_QSS", "PiPWidget", "CameraWorker", "TrainWorker"]
