"""Import smoke test."""

import nngeom as ng


def test_import_and_version() -> None:
    assert ng.__version__
    assert callable(ng.analyze)
    assert "effective_rank" in ng.list_metrics()
