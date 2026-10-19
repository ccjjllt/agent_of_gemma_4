"""P012 dev criterion: RecipeReference.matches supports the &: consumer pattern."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ampersand_colon_matches_consumer():
    from conan.api.model import RecipeReference

    ref = RecipeReference("pkg", "1.0", None, None, None)
    assert ref.matches("&:", True) is True
    assert ref.matches("&", True) is True
