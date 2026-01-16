# from PySide6.QtCore import QObject
# from abc import ABCMeta, abstractmethod
# class MetaBaseTab(ABCMeta, type(QObject)):
#    """A metaclass combining ABCMeta and Qt's metaclass"""
#    pass
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QWidget


class BaseReinforcementLearningTab(QWidget):  # (QWidget, metaclass=MetaBaseTab)
    """Base class for all Reinforcement Learning parameter tabs"""

    paramsChanged = Signal()

    def __init__(self):
        super().__init__()
        self.params = {}
        self.widgets = {}

    def add_param_widget(self, layout, label, widget, param_name):
        """Helper to add a parameter widget to layout"""
        layout.addRow(QLabel(label), widget)
        self.widgets[param_name] = widget

    def get_params(self):
        """Override in subclasses to return parameters"""
        return {}
