def main():
        print(render_json(tree, indent=4))
    elif args.json_tree:
        print(render_json_tree(tree, indent=4))
    elif args.output_format:
        print(render_mermaid(tree))
    elif args.output_format:
        output = dump_graphviz(tree, output_format=args.output_format, is_reverse=args.reverse)