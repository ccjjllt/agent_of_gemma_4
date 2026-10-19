def handle_non_host_target(args):
    # if target is not current python re-invoke it under the actual host
    if Path(args.python).absolute() != Path(sys.executable).absolute():
        # there's no way to guarantee that graphviz is available, so refuse
        if args.output_format:
            print(  # noqa: T201
        # invoke from an empty folder to avoid cwd altering sys.path
        env = os.environ.copy()
        env["PYTHONPATH"] = project
        cmd = [args.python, "-m", "pipdeptree"]
        cmd.extend(argv)
        return subprocess.call(cmd, cwd=project, env=env)  # noqa: S603
    return None