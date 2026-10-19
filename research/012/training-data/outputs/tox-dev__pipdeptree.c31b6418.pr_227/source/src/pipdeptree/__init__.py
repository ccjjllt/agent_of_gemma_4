def get_parser():
        "--mermaid",
        action="store_true",
        default=False,
        help=("Display dependency tree as a Maermaid graph. " "This option overrides all other options."),
    )
    parser.add_argument(
        "--graph-output",