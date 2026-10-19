from collections.abc import Mapping
from importlib import import_module
from itertools import chain
from pip._vendor import pkg_resources


def aux(node, parent=None, cur_chain=None):
    return json.dumps([aux(p) for p in nodes], indent=indent)


def dump_graphviz(tree, output_format="dot", is_reverse=False):
    """Output dependency graph as one of the supported GraphViz output formats.
    """


def get_parser():
             "This option overrides all other options (except --json)."
         ),
     )
    parser.add_argument(
         "--graph-output",
         dest="output_format",
    )


def main():
    if args.json:
         print(render_json(tree, indent=4))
    elif args.json_tree:
         print(render_json_tree(tree, indent=4))
    elif args.output_format:
         output = dump_graphviz(tree, output_format=args.output_format, is_reverse=args.reverse)
         print_graphviz(output)