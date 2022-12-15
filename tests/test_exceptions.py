"""Tests for exceptions and requests"""
import pytest

import pynldas2 as nldas
from pynldas2 import NLDASServiceError

def test_invalid_method():
    with pytest.raises(NLDASServiceError) as ex:
        _ = nldas.get_bycoords()
    assert "GET" in str(ex.value)
