{title="a.py"}
        IntList = list[int]
        OuterDict = dict[str, 'IntList']
        ```

        ```python {test="skip" title="b.py"}
        from a import OuterDict

        from pydantic import TypeAdapter

        IntList = int  # replaces the symbol the forward reference is looking for
        v = TypeAdapter(OuterDict)
        v({'x': 1})  # should fail but doesn't