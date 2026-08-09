import ragci


def test_package_exposes_a_version():
    assert isinstance(ragci.__version__, str)
    assert ragci.__version__.count(".") == 2
