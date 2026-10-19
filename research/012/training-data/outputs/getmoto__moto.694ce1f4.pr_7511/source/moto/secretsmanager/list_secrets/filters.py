import re
from typing import TYPE_CHECKING, Iterator, List, Union

if TYPE_CHECKING:
    from ..models import FakeSecret


def name_filter(secret: "FakeSecret", names: List[str]) -> bool:
    return _matcher(names, [secret.name])


def description_filter(secret: "FakeSecret", descriptions: List[str]) -> bool:
    if not secret.description:
        return False
    # The documentation states that this search uses `Prefix match`
    # But actual testing determines that it uses the same approach to the `all_filter`:
    # 'Breaks the filter value string into words and then searches all attributes for matches.'
    return _matcher(descriptions, [secret.description])  # type: ignore


def tag_key(secret: "FakeSecret", tag_keys: List[str]) -> bool:
    if not secret.tags:
        return False
    return _matcher(tag_keys, [tag["Key"] for tag in secret.tags])


def tag_value(secret: "FakeSecret", tag_values: List[str]) -> bool:
    if not secret.tags:
        return False
    return _matcher(tag_values, [tag["Value"] for tag in secret.tags])


def filter_all(secret: "FakeSecret", values: List[str]) -> bool:
    attributes = [secret.name]
    if secret.description:
        attributes.append(secret.description)
    if secret.tags:
        attributes += [tag["Key"] for tag in secret.tags] + [
            tag["Value"] for tag in secret.tags
        ]

    return _matcher(values, attributes)  # type: ignore


def _matcher(patterns: List[str], strings: List[str]) -> bool:
    for pattern in [p for p in patterns if p.startswith("!")]:
        for string in strings:
            if _match_pattern(pattern[1:], string):
                return False
    for pattern in [p for p in patterns if not p.startswith("!")]:
        for string in strings:
            if _match_pattern(pattern, string):
                return True
    return False


def _match_pattern(pattern: str, value: str) -> bool:
    for word in pattern.split(" "):
        if word not in value:
            return False
    return True


def split_words(s: str) -> List[str]:
    """
    Secrets are split by special characters first (/, +, _, etc)
    Partial results are then split again by UpperCasing
    """
    special_chars = ["/", "-", "_", "+", "=", ".", "@"]

    if s in special_chars:
        # Special case: this does not return any values
        return []

    for char in special_chars:
        if char in s:
            others = special_chars.copy()
            others.remove(char)
            contains_other = any([c in s for c in others])
            if contains_other:
                # Secret contains two different characters, i.e. my/secret+value
                # Values like this will not be split
                return [s]
            else:
                return list(split_by_uppercase(s.split(char)))
    return list(split_by_uppercase(s))


def split_by_uppercase(s: Union[str, List[str]]) -> Iterator[str]:
    """
    Split a string into words. Words are recognized by upper case letters, i.e.:
    test   -> [test]
    MyTest -> [My, Test]
    """
    if isinstance(s, str):
        for x in re.split(r"([^a-z][a-z]+)", s):
            if x:
                yield x.strip()
    else:
        for word in s:
            yield from split_by_uppercase(word)