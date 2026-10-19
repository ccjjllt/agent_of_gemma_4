"""
    pygments.styles
    ~~~~~~~~~~~~~~~

    Contains built-in styles.

    :copyright: Copyright 2006-2024 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from pygments.plugin import find_plugin_styles
from pygments.util import ClassNotFound
from pygments.styles._mapping import STYLES

#: A dictionary of built-in styles, mapping style names to
#: ``'submodule::classname'`` strings.
#: This list is deprecated. Use `pygments.styles.STYLES` instead
STYLE_MAP = {
    'default':  'default::DefaultStyle',
    'emacs':    'emacs::EmacsStyle',
    'colorful': 'colorful::ColorfulStyle',
    'autumn':   'autumn::AutumnStyle',
    'murphy':   'murphy::MurphyStyle',
    'manni':    'manni::ManniStyle',
    'monokai':  'monokai::MonokaiStyle',
    'perldoc':  'perldoc::PerldocStyle',
    'pastie':   'pastie::PastieStyle',
    'borland':  'borland::BorlandStyle',
    'trac':     'trac::TracStyle',
    'native':   'native::NativeStyle',
    'fruity':   'fruity::FruityStyle',
    'bw':       'bw::BlackWhiteStyle',
    'vim':      'vim::VimStyle',
    'vs':       'vs::VisualStudioStyle',
    'tango':    'tango::TangoStyle',
    'rrt':      'rrt::RrtStyle',
    'xcode':    'xcode::XcodeStyle',
    'igor':     'igor::IgorStyle',
    'paraiso-light': 'paraiso_light::ParaisoLightStyle',
    'lovelace': 'lovelace::LovelaceStyle',
    'algol':    'algol::AlgolStyle',
    'algol_nu': 'algol_nu::Algol_NuStyle',
    'arduino':  'arduino::ArduinoStyle',
    'paraiso-dark': 'paraiso_dark::ParaisoDarkStyle',
    'sas':         'sas::SasStyle',
    'staroffice' : 'staroffice::StarofficeStyle',
    'stata':       'stata_light::StataLightStyle',
    'stata-dark':  'stata_dark::StataDarkStyle',
    'inkpot':      'inkpot::InkPotStyle',
    'zenburn': 'zenburn::ZenburnStyle',
    'gruvbox-dark': 'gruvbox::GruvboxDarkStyle',
    'gruvbox-light': 'gruvbox::GruvboxLightStyle',
    'dracula': 'dracula::DraculaStyle',
    'one-dark': 'onedark::OneDarkStyle',
    'lilypond' : 'lilypond::LilyPondStyle',
    'nord': 'nord::NordStyle',
    'nord-darker': 'nord::NordDarkerStyle',
    'github-dark': 'gh_dark::GhDarkStyle'
}

#: Internal reverse mapping to make `get_style_by_name` more efficient
_STYLE_NAME_TO_MODULE_MAP = {v[1]: (v[0], k) for k, v in STYLES.items()}


def get_style_by_name(name):
    """
    Return a style class by its short name. The names of the builtin styles
    are listed in :data:`pygments.styles.STYLE_MAP`.

    Will raise :exc:`pygments.util.ClassNotFound` if no style of that name is
    found.
    """
    if name in _STYLE_NAME_TO_MODULE_MAP:
        mod, cls = _STYLE_NAME_TO_MODULE_MAP[name]
        builtin = "yes"
    else:
        for found_name, style in find_plugin_styles():
            if name == found_name:
                return style
        # perhaps it got dropped into our styles package
        builtin = ""
        mod = 'pygments.styles.' + name
        cls = name.title() + "Style"

    try:
        mod = __import__(mod, None, None, [cls])
    except ImportError:
        raise ClassNotFound(f"Could not find style module {mod!r}" +
                            (builtin and ", though it should be builtin")
                            + ".")
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ClassNotFound(f"Could not find style class {cls!r} in style module.")


def get_all_styles():
    """Return a generator for all styles by name, both builtin and plugin."""
    for v in STYLES.values():
        yield v[1]
    for name, _ in find_plugin_styles():
        yield name