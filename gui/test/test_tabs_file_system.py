
import pytest
from gui.src.tabs.file_system.fs_update import FileSystemUpdateTab
from gui.src.tabs.file_system.fs_delete import FileSystemDeleteTab
from gui.src.tabs.file_system.fs_cryptography import FileSystemCryptographyTab

def test_fs_update_tab(qapp):
    tab = FileSystemUpdateTab()
    # Usually these tabs are simple forms or buttons
    assert tab is not None

def test_fs_delete_tab(qapp):
    tab = FileSystemDeleteTab()
    assert tab is not None

def test_fs_crypto_tab(qapp):
    tab = FileSystemCryptographyTab()
    assert tab is not None
