"""P012 dev criterion: parsing does not raise NameError for cgi."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "p012_lib"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEED = """<rss version="2.0"><channel><title>T</title><link>http://e.com/</link>""" \
    """<description>D</description></channel></rss>"""


def test_parse_no_cgi_name_error():
    import feedparser

    result = feedparser.parse(FEED)
    assert result["bozo"] is False, result.get("bozo_exception")
