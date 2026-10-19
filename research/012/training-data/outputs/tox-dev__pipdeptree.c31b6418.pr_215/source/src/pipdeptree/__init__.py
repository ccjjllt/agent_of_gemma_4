import argparse
import inspect
import json
import os

def filter(self, include, exclude):
         m = {}
         seen = set()
         for node in self._obj.keys():
             if node.key in exclude:
                 continue
             if include is None or node.key in include:
                 stack.append(node)
             while True:
                 if len(stack) > 0:
                     n = stack.pop()
                     cldn = [c for c in self._obj[n] if c.key not in exclude]
                     m[n] = cldn
                     seen.add(n.key)
                     for c in cldn:

def get_parser():
    parser.add_argument(
         "-p",
         "--packages",
         help="Comma separated list of select packages to show in the output. If set, --all will be ignored.",
    )
    parser.add_argument(
         "-e",
         "--exclude",
         help="Comma separated list of select packages to exclude from the output. If set, --all will be ignored.",
         metavar="PACKAGES",
    )
    parser.add_argument(