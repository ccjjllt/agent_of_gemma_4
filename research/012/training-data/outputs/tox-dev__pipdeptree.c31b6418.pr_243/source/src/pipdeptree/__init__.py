import argparse
import fnmatch
import inspect

from collections.abc import Mapping
from import_module import import_module
from itertools import chain
from textwrap import dedent

from pip._vendor import pkg_resources


def sorted_tree(tree):
    pass


def guess_version(pkg_key, default="?"):
    """Guess the version of a pkg when pip doesn't provide it
    :param str pkg_key: key of the package
    :param str default: default version to return if unable to find
    """
    m = import_module(pkg_key)
    v = getattr(m, "__version__", default)
    if inspect.ismodule(v):
        return getattr(v, "__version__", default)
    else:
        return v


def frozen_req_from_dist(dist):
    pass


class Package:
    """
    Wrapper class for package objects to standardize the interface.
    Provides an interface for displaying a dependency tree.
    Designed to be subclassed by ReqPackage and DistPackage.
    """

    def __init__(self, obj):
        self._obj = obj
        self.project_name = obj.project_name
        self.key = obj.key

    def render_as_root(self, frozen):  # noqa: U100
        return NotImplementedError

    def render_as_branch(self, frozen):  # noqa: U100
        return NotImplementedError

    def render(self, parent=None, frozen=False):
        if not parent:
            return self.render_as_root(frozen)
        else:
            return self.render_as_branch(frozen)

    @staticmethod
    def frozen_repr(obj):
        return repr(obj)

    def __getattr__(self, key):
        return getattr(self._obj, key)

    def __repr__(self):
        return f'<{self.__class__.__name__}("{self.key}")>'

    def __lt__(self, rhs):
        return self.key < rhs.key


class DistPackage(Package):
    """
    Wrapper class for pkg_resources.Distribution instances
    :param obj: pkg_resources.Distribution to wrap over
    :param req: optional ReqPackage object to associate this DistPackage with. This is useful for displaying the tree
        in reverse
    """

    def __init__(self, obj, req=None):
        super().__init__(obj)
        self.version_spec = None
        self.req = req

    def render_as_root(self, frozen):
        if not frozen:
            return f"{self.project_name}=={self.version}"
        else:
            return self.__class__.frozen_repr(self._obj)

    def render_as_branch(self, frozen):
        assert self.req is not None
        if not frozen:
            parent_ver_spec = self.req.version_spec
            parent_str = self.req.project_name
            if parent_ver_spec:
                parent_str += parent_ver_spec
            return f"{self.project_name}=={self.version} [requires: {parent_str}]"
        else:
            return self.render_as_root(frozen)

    def as_requirement(self):
        """Return a ReqPackage representation of this DistPackage"""
        return ReqPackage(self._obj.as_requirement(), dist=self)

    def as_parent_of(self, req):
        pass

    def as_dict(self):
        pass


class ReqPackage(Package):
    """
    Wrapper class for Requirements instance
    :param obj: The `Requirements` instance to wrap over
    :param dist: optional `pkg_resources.Distribution` instance for this requirement
    """

    UNKNOWN_VERSION = "?"
    def __init__(self, obj, dist=None):
        super().__init__(obj)
        self.dist = dist

    def is_missing(self):
        return self.installed_version == self.UNKNOWN_VERSION

    def is_conflicting(self):
        """If installed version conflicts with required version"""
        if self.installed_version == self.UNKNOWN_VERSION:
            return True
        return False

    def render_as_root(self, frozen):
        if not frozen:
            return f"{self.project_name}=={self.installed_version}"
        elif self.dist:
            return self.__class__.frozen_repr(self.dist._obj)
        else:
            return self.project_name

    def render_as_branch(self, frozen):
        if not frozen:
            req_ver = self.version_spec if self.version_spec else "Any"
            return f"{self.project_name} [required: {req_ver}, installed: {self.installed_version}]"
        else:
            return self.render_as_root(frozen)

    def as_dict(self):
        return {}


class PackageDAG:
    @classmethod
    def from_pkgs(cls, pkgs):
        m = {p: [ReqPackage(r, None) for r in p.requires()] for p in pkgs}
        return cls(m)

    def __init__(self, m):
        """Initialize the PackageDAG object
        :param dict m: dict of node objects (refer class docstring)
        :returns: None
        """
        self._obj = m

    def get_node_as_parent(self, node_key):
        pass

    def get_children(self, node_key):
        node = self.get_node_as_parent(node_key)
        return self._obj[node] if node else []

    def filter(self, include, exclude):
        """
        Filters nodes in a graph by given parameters
        If a node is included, then all it's children are also included.
        """
        if include:
            include = {s.lower() for s in include}
        if exclude:
            exclude = {s.lower() for s in exclude}
        else:
            exclude = set()
        if include and exclude:
            assert not (include & exclude)
        from collections import deque
        stack = deque()
        m = {}
        seen = set()
        for node in self._obj.keys():
            if any(fnmatch.fnmatch(node.key, e) for e in exclude):
                continue
            if include is None or any(fnmatch.fnmatch(node.key, i) for i in include):
                m[node] = self._obj[node]
        return m

    def reverse(self):
        m = {}
        for k, vs in self._obj.items():
            for v in vs:
                try:
                    node = [p for p in m.keys() if p.key == v.key][0]
                except IndexError:
                    node = v
                m.setdefault(node, []).append(k.as_parent_of(v))
        return PackageDAG(dict(m))

    def __getitem__(self, *args):
        return self._obj.__getitem__(*args)

    def __iter__(self):
        return self._obj.__iter__()

    def __len__(self):
        return len(self._obj)


class ReversedPackageDAG(PackageDAG):
    """Representation of Package dependencies in the reverse order.
    Similar to it's super class `PackageDAG`, the underlying datastructure is a dict, but here the keys are expected to
    be of type `ReqPackage` and each item in the values of type `DistPackage`.
    """

    def reverse(self):
        """
        Reverse the already reversed DAG to get the PackageDAG again
        :returns: reverse of the reversed DAG
        :rtype: PackageDAG
        """
        m = {}
        for k, vs in self._obj.items():
            for v in vs:
                try:
                    node = [p for p in m.keys() if p.key == v.key][0]
                except IndexError:
                    node = v.as_parent_of(None)
                m.setdefault(node, []).append(k)
        return PackageDAG(dict(m))


def render_text(tree, max_depth, list_all=True, frozen=False):
    """Print tree as text on console
    :param dict tree: the package tree
    :param bool list_all: whether to list all the pgks at the root level or only those that are the sub-dependencies
    :param bool frozen: whether to use frozen representation
    """
    _render_text_with_unicode(tree, list(tree._obj.keys()), max_depth, frozen)


def _render_text_with_unicode(tree, nodes, max_depth, frozen):
    use_bullets = not frozen

    def aux(
        node,
        parent=None,
        indent=0,
        cur_chain=None,
        prefix="",
        depth=0,
        has_grand_parent=False,
        is_last_child=False,
        parent_is_last_child=False,
    ):
        cur_chain = cur_chain or []
        node_str = node.render(parent, frozen)
        result = [(" " * indent) + prefix + node_str]
        next_indent = indent + 2
        next_prefix = prefix
        for c in tree.get_children(node.key):
            result.extend(aux(c, node, indent=next_indent, cur_chain=cur_chain + [c.project_name], prefix=next_prefix, depth=depth + 1, has_grand_parent=(parent is not None), is_last_child=False, parent_is_last_child=is_last_child))
        return result

    lines = chain.from_iterable([aux(p) for p in nodes])
    print("\n".join(lines))


def _render_text_without_unicode(tree, nodes, max_depth, frozen):
    def aux(node, parent=None, indent=0, cur_chain=None, depth=0):
        cur_chain = cur_chain or []
        node_str = node.render(parent, frozen)
        if depth > 0:
            node_str = (" " * indent) + node_str
        result = [node_str]
        for c in tree.get_children(node.key):
            result.extend(aux(c, parent=node, indent=indent + 2, cur_chain=cur_chain + [c.project_name], depth=depth + 1))
        return result

    lines = chain.from_iterable([aux(p) for p in nodes])
    print("\n".join(lines))


def render_json(tree, indent):
    tree = tree.sort() if hasattr(tree, "sort") else tree
    import json
    return json.dumps(
        [{"package": k.as_dict(), "dependencies": [v.as_dict() for v in vs]} for k, vs in tree.items()], indent=indent
    )


def render_json_tree(tree, indent):
    tree = tree.sort() if hasattr(tree, "sort") else tree
    from itertools import chain
    branch_keys = {r.key for r in chain.from_iterable(tree.values())}
    nodes = [p for p in tree.keys() if p.key not in branch_keys]

    def aux(node, parent=None, cur_chain=None):
        if cur_chain is None:
            cur_chain = [node.project_name]
        d = node.as_dict()
        if "installed_version" in d and d["installed_version"] == "?":
            d["required_version"] = d["installed_version"]
        d["dependencies"] = [
            aux(c, parent=node, cur_chain=cur_chain + [c.project_name])
            for c in tree.get_children(node.key)
            if c.project_name not in cur_chain
        ]
        return d

    import json
    return json.dumps([aux(p) for p in nodes], indent=indent)


def render_mermaid(tree) -> str:
    """Produce a Mermaid flowchart from the dependency graph.
    :param dict tree: dependency graph
    """
    nodes = set()
    edges = set()

    def mermaid_id(key: str) -> str:
        return "node_" + key.replace(" ", "_")

    if isinstance(tree, ReversedPackageDAG):
        for package, reverse_dependencies in tree.items():
            package_label = "\\n".join((package.project_name, "(missing)" if package.is_missing else package.installed_version))
            package_key = mermaid_id(package.key)
            nodes.add(f'{package_key}["{package_label}"]')
            for reverse_dependency in reverse_dependencies:
                edge_label = ""
                reverse_dependency_key = mermaid_id(reverse_dependency.key)
                edges.add(f'{package_key} -- "{edge_label}" --> {reverse_dependency_key}')
    else:
        for package, dependencies in tree.items():
            package_label = "\\n".join((package.project_name, package.version))
            package_key = mermaid_id(package.key)
            nodes.add(f'{package_key}["{package_label}"]')
            for dependency in dependencies:
                dependency_key = mermaid_id(dependency.key)
                edges.add(f'{package_key} -- "{dependency.version}" --> {dependency_key}')
    indent = "    "
    output = (
        f"""\
flowchart TD
{indent}classDef missing stroke-dasharray: 5
"""
    )
    output += indent + "\n".join(sorted(nodes)) + "\n"
    output += indent + "\n".join(sorted(edges)) + "\n"
    output += indent + "\n"
    return output


def dump_graphviz(tree, output_format="dot", is_reverse=False):
    """Output dependency graph as one of the supported GraphViz output formats.
    :param dict tree: dependency graph
    :param string output_format: output format
    :param bool is_reverse: whether the tree is reversed
    :returns: graphviz output
    """
    import sys
    try:
        from graphviz import Digraph
    except ImportError:
        print("graphviz is not available, but necessary for the output option. Please install it.", file=sys.stderr)
        sys.exit(1)
    try:
        from graphviz import backend, parameters
    except ImportError:
        pass
    valid_formats = backend.FORMATS
    if output_format not in valid_formats:
        print(f"{output_format} is not a supported output format.", file=sys.stderr)
        print(f"Supported formats are: {', '.join(sorted(valid_formats))}", file=sys.stderr)
        sys.exit(1)
    graph = Digraph(format=output_format)
    return "".join([tuple(graph)[0]] + sorted(graph.body) + [graph._tail])


def print_graphviz(dump_output):
    import sys, os
    if hasattr(dump_output, "encode"):
        print(dump_output)
    else:
        with os.fdopen(sys.stdout.fileno(), "wb") as bytestream:
            bytestream.write(dump_output)


def conflicting_deps(tree):
    return {}


def render_conflicts_text(conflicts):
    if conflicts:
        import sys
        print("Warning!!! Possibly conflicting dependencies found:", file=sys.stderr)
        pkgs = sorted(conflicts.keys())
        for p in pkgs:
            pkg = p.render_as_root(False)
            print(f"* {pkg}", file=sys.stderr)
            for req in conflicts[p]:
                req_str = req.render_as_branch(False)
                print(f" - {req_str}", file=sys.stderr)


def cyclic_deps(tree):
    """
    Return cyclic dependencies as list of tuples
    :param PackageDAG tree: package tree/dag
    :returns: list of tuples representing cyclic dependencies
    """
    return []


def render_cycles_text(cycles):
    if cycles:
        import sys
        print("Warning!! Cyclic dependencies found:", file=sys.stderr)
        cycles = sorted(cycles, key=lambda xs: xs[1].key)
        for a, b, c in cycles:
            print(f"* {a.project_name} => {b.project_name} => {c.project_name}", file=sys.stderr)


def get_parser():
    import sys
    parser = argparse.ArgumentParser("pipdeptree")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python to use to look for packages in it (default: where" " installed)",
    )
    parser.add_argument("-a", "--all", action="store_true", help="list all deps at top level")
    parser.add_argument(
        "-l",
        "--local-only",
        action="store_true",
        help="If in a virtualenv that has global access " "do not show globally installed packages",
    )
    parser.add_argument("-u", "--user-only", action="store_true", help="Only show installations in the user site dir")
    parser.add_argument(
        "--warn",
        default="silent",
        choices=["silent", "fail"],
        help="Do nothing or exit with status 1 if conflicts or cycles are found",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Use `pip freeze` format for the output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output dependency tree as JSON",
    )
    parser.add_argument(
        "--json-tree",
        action="store_true",
        help="Output dependency tree as JSON tree",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        default=False,
        help=("Display dependency tree as a Mermaid graph. " "This option overrides all other options."),
    )
    parser.add_argument(
        "--graph-output",
        default="dot",
        dest="output_format",
        help="GraphViz output rendition: dot, png, etc",
    )
    parser.add_argument(
        "--packages",
        default="",
        help="Comma separated package names to filter the output for",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma separated package names to exclude the output for",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Displayed dependency tree depth, 0 implies it is unlimited",
    )
    return parser


def _get_args():
    parser = get_parser()
    return parser.parse_args()


def handle_non_host_target(args):
    import sys
    import os
    import tempfile
    import shutil
    import inspect
    import subprocess
    of_python = os.path.abspath(args.python)
    if of_python != os.path.abspath(sys.executable):
        if args.output_format:
            print("graphviz functionality is not supported when querying" " non-host python", file=sys.stderr)
            sys.exit(1)
        argv = sys.argv[1:]
        for py_at, value in enumerate(argv):
            if os.path.isabs(value) and value == sys.executable:
                argv[py_at] = of_python
        main_file = inspect.getsourcefile(sys.modules[__name__])
        with tempfile.TemporaryDirectory() as project:
            dest = os.path.join(project, "pipdeptree")
            shutil.copytree(os.path.dirname(main_file), dest)
            env = os.environ.copy()
            env["PYTHONPATH"] = project
            cmd = [of_python, "-m", "pipdeptree"]
            cmd.extend(argv)
            return subprocess.call(cmd, cwd=project, env=env)
    return None


def get_installed_distributions(local_only=False, user_only=False):
    try:
        from pip._internal.metadata import pkg_resources
    except ImportError:
        import pkg_resources
    dists = pkg_resources.Environment.from_paths(None).iter_installed_distributions(local_only=local_only, skip=(), user_only=user_only)
    return [d._dist for d in dists]


def main():
    import sys
    args = _get_args()
    result = handle_non_host_target(args)
    if result is not None:
        return result
    pkgs = get_installed_distributions(local_only=args.local_only, user_only=args.user_only)
    tree = PackageDAG.from_pkgs(pkgs)
    is_text_output = not any([args.json, args.json_tree, args.output_format])
    return_code = 0
    if args.json:
        print(render_json(tree, indent=4))
    elif args.json_tree:
        print(render_json_tree(tree, indent=4))
    elif args.mermaid:
        print(render_mermaid(tree))
    elif args.output_format:
        output = dump_graphviz(tree, output_format=args.output_format, is_reverse=args.reverse)
        print_graphviz(output)
    else:
        render_text(tree, args.depth, args.all, args.freeze)
    return return_code


if __name__ == "__main__":
    import sys
    sys.exit(main())