import importlib

import pytest

MODULES = ["load", "clean", "ballfilter", "features", "deadball", "decode", "model", "tune", "pipeline",
           "submission", "external.idsse", "external.metrica", "external.bench"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"possession.{name}")
