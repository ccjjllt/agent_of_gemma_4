"""P012 dev criterion: convert_to_idn importable; basic parse has bozo False."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "p012_lib"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEED = """<rss version="2.0"><channel><title>T</title><link>http://e.com/</link>""" \
    """<description>D</description></channel></rss>"""


def test_idn_import_and_parse():
    from feedparser.urls import convert_to_idn

    assert callable(convert_to_idn)
    import feedparser

    result = feedparser.parse(FEED)
    assert result["bozo"] is False, result.get("bozo_exception")
