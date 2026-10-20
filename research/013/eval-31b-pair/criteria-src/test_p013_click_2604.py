"""P013 dev criterion: @command() strips _command/_cmd/_group/_grp suffixes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_command_name_strips_suffix():
    from click import command

    @command()
    def init_data_command():
        pass

    assert init_data_command.name == "init-data"

    @command()
    def init_data_cmd():
        pass

    assert init_data_cmd.name == "init-data"

    @command()
    def init_data_group():
        pass

    assert init_data_group.name == "init-data"

    @command()
    def init_data_grp():
        pass

    assert init_data_grp.name == "init-data"
