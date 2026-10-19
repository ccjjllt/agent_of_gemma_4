"""Jinja2 extensions."""

from __future__ import annotations

import json
import string
import uuid
from secrets import choice
from typing import TYPE_CHECKING, Any, Iterable

from jinja2.ext import Extension
from slugify import slugify as pyslugify
from slugify.slugify import DEFAULT_SEPARATOR

if TYPE_CHECKING:
    import re

    from jinja2.parser import Parser


class JsonifyExtension(Extension):
    """Jinja2 extension to convert a Python object to JSON."""

    def __init__(self, environment: Environment) -> None:
        """Initialize the extension with the given environment."""
        super().__init__(environment)

        def jsonify(obj: Any, indent: int = 4) -> str:
            return json.dumps(obj, sort_keys=True, indent=indent)

        environment.filters['jsonify'] = jsonify


class RandomStringExtension(Extension):
    """Jinja2 extension to create a random string."""

    def __init__(self, environment: Environment) -> None:
        """Jinja2 Extension Constructor."""
        super().__init__(environment)

        def random_ascii_string(length: int, punctuation: bool = False) -> str:
            if punctuation:
                corpus = f'{string.ascii_letters}{string.punctuation}'
            else:
                corpus = string.ascii_letters
            return "".join(choice(corpus) for _ in range(length))

        environment.globals.update(random_ascii_string=random_ascii_string)


class SlugifyExtension(Extension):
    """Jinja2 Extension to slugify string."""

    def __init__(self, environment: Environment) -> None:
        """Jinja2 Extension constructor."""
        super().__init__(environment)

        def slugify(
            value: str,
            entities: bool = True,
            decimal: bool = True,
            hexadecimal: bool = True,
            max_length: int = 0,
            word_boundary: bool = False,
            separator: str = DEFAULT_SEPARATOR,
            save_order: bool = False,
            stopwords: Iterable[str] = (),
            regex_pattern: re.Pattern[str] | str | None = None,
            lowercase: bool = True,
            replacements: Iterable[Iterable[str]] = (),
            allow_unicode: bool = False,
        ) -> str:
            """Slugifies the value."""
            return pyslugify(
                value,
                entities,
                decimal,
                hexadecimal,
                max_length,
                word_boundary,
                separator,
                save_order,
                stopwords,
                regex_pattern,
                lowercase,
                replacements,
                allow_unicode,
            )

        environment.filters['slugify'] = slugify


class UUIDExtension(Extension):
    """Jinja2 Extension to generate uuid4 string."""

    def __init__(self, environment: Environment) -> None:
        """Jinja2 Extension constructor."""
        super().__init__(environment)

        def uuid4() -> str:
            """Generate UUID4."""
            return str(uuid.uuid4())

        environment.globals.update(uuid4=uuid4)