"""This module contains related classes and functions for serialization."""

from __future__ import annotations

import dataclasses
from functools import partial, partialmethod
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, TypeVar, overload

from pydantic_core import PydanticUndefined, core_schema
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler, WhenUsed
from typing_extensions import TypeAlias

from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler


@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class PlainSerializer:
    """Plain serializers use a function to modify the output of serialization.

    Attributes:
        func: The serializer function.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.
    """

    func: core_schema.SerializerFunction
    return_type: Any = PydanticUndefined
    when_used: WhenUsed = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Gets the Pydantic core schema.

        Args:
            source_type: The source type.
            handler: The `GetCoreSchemaHandler` instance.

        Returns:
            The Pydantic core schema.
        """
        schema = handler(source_type)
        if self.return_type is not PydanticUndefined:
            return_type = self.return_type
        else:
            try:
                # Do not pass in globals as the function could be defined in a different module.
                # Instead, let `get_callable_return_type` infer the globals to use, but still pass
                # in locals that may contain a parent/rebuild namespace:
                return_type = _decorators.get_callable_return_type(
                    self.func,
                    localns=handler._get_types_namespace().locals,
                )
            except NameError as e:
                raise PydanticUndefinedAnnotation.from_name_error(e) from e

        return_schema = None if return_type is PydanticUndefined else handler.generate_schema(return_type)
        schema['serialization'] = core_schema.plain_serializer_function_ser_schema(
            function=self.func,
            info_arg=_decorators.inspect_annotated_serializer(self.func, 'plain'),
            return_schema=return_schema,
            when_used=self.when_used,
        )
        return schema


@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class WrapSerializer:
    """Wrap serializers receive the raw inputs along with a handler function that applies the standard serialization
    logic, and can modify the resulting value before returning it as the final output of serialization.

    Attributes:
        func: The serializer function to be wrapped.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.
        when_used: The serializer function to be wrapped.
    """

    func: core_schema.WrapSerializerFunction
    return_type: Any = PydanticUndefined
    when_used: WhenUsed = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """This method is used to get the Pydantic core schema of the class.

        Args:
            source_type: Source type.
            handler: Core schema handler.

        Returns:
            The generated core schema of the class.
        """
        schema = handler(source_type)
        if self.return_type is not PydanticUndefined:
            return_type = self.return_type
        else:
            try:
                # Do not pass in globals as the function could be defined in a different module.
                # Instead, let `get_callable_return_type` infer the globals to use, but still pass
                # in locals that may contain a parent/rebuild namespace:
                return_type = _decorators.get_callable_return_type(
                    self.func,
                    localns=handler._get_types_namespace().locals,
                )
            except NameError as e:
                raise PydanticUndefinedAnnotation.from_name_error(e) from e

        return_schema = None if return_type is PydanticUndefined else handler.generate_schema(return_type)
        schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(
            function=self.func,
            info_arg=_decorators.inspect_annotated_serializer(self.func, 'wrap'),
            return_schema=return_schema,
            when_used=self.when_used,
        )
        return schema


if TYPE_CHECKING:
    _Partial: TypeAlias = 'partial[Any] | partialmethod[Any]'

    FieldPlainSerializer: TypeAlias = 'core_schema.SerializerFunction | _Partial'
    """A field serializer method or function in `plain` mode."""

    FieldWrapSerializer: TypeAlias = 'core_schema.WrapSerializerFunction | _Partial'
    """A field serializer method or function in `wrap` mode."""

    FieldSerializer: TypeAlias = 'FieldPlainSerializer | FieldWrapSerializer'
    """A field serializer method or function."""

    _FieldPlainSerializerT = TypeVar('_FieldPlainSerializerT', bound=FieldPlainSerializer)
    _FieldWrapSerializerT = TypeVar('_FieldWrapSerializerT', bound=FieldWrapSerializer)


@overload
def field_serializer(
    field: str,
    /,
    *fields: str,
    mode: Literal['wrap'],
    return_type: Any = ...,
    when_used: WhenUsed = ...,
    check_fields: bool | None = ...,
) -> Callable[[_FieldWrapSerializerT], _FieldWrapSerializerT]: ...

@overload
def field_serializer(
    field: str,
    /,
    *fields: str,
    mode: Literal['plain'] = ...,
    return_type: Any = ...,
    when_used: WhenUsed = ...,
    check_fields: bool | None = ...,
) -> Callable[[_FieldPlainSerializerT], _FieldPlainSerializerT]: ...


def field_serializer(
    *fields: str,
    mode: Literal['plain', 'wrap'] = 'plain',
    return_type: Any = PydanticUndefined,
    when_used: WhenUsed = 'always',
    check_fields: bool | None = None,
) -> (
    Callable[[_FieldWrapSerializerT], _FieldWrapSerializerT]
    | Callable[[_FieldPlainSerializerT], _FieldPlainSerializerT]
):
    """Decorator that enables custom field serialization.

    See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.

    Four signatures are supported:

    - `(self, value: Any, info: FieldSerializationInfo)`
    - `(self, value: Any, nxt: SerializerFunctionWrapHandler, info: FieldSerializationInfo)`
    - `(value: Any, info: SerializationInfo)`
    - `(value: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo)`

    Args:
        fields: Which field(s) the method should be called on.
        mode: The serialization mode.

            - `plain` means the function will be called instead of the default serialization logic,
            - `wrap` means the function will be called with an argument to optionally call the
               default serialization logic.
        return_type: Optional return type for the function, if omitted it will be inferred from the type annotation.
        when_used: Determines the serializer will be used for serialization.
        check_fields: Whether to check that the fields actually exist on the model.

    Returns:
        The decorator function.
    """

    def dec(f: FieldSerializer) -> _decorators.PydanticDescriptorProxy[Any]:
        dec_info = _decorators.FieldSerializerDecoratorInfo(
            fields=fields,
            mode=mode,
            return_type=return_type,
            when_used=when_used,
            check_fields=check_fields,
        )
        return _decorators.PydanticDescriptorProxy(f, dec_info)  # pyright: ignore[reportArgumentType]

    return dec  # pyright: ignore[reportReturnType]


if TYPE_CHECKING:
    # The first argument in the following callables represent the `self` type:

    ModelPlainSerializerWithInfo: TypeAlias = Callable[[Any, SerializationInfo], Any]
    """A model serializer method with the `info` argument, in `plain` mode."""

    ModelPlainSerializerWithoutInfo: TypeAlias = Callable[[Any], Any]
    """A model serializer method without the `info` argument, in `plain` mode."""

    ModelPlainSerializer: TypeAlias = 'ModelPlainSerializerWithInfo | ModelPlainSerializerWithoutInfo'
    """A model serializer method in `plain` mode."""

    ModelWrapSerializerWithInfo: TypeAlias = Callable[[Any, SerializerFunctionWrapHandler, SerializationInfo], Any]
    """A model serializer method with the `info` argument, in `wrap` mode."""

    ModelWrapSerializerWithoutInfo: TypeAlias = Callable[[Any, SerializerFunctionWrapHandler], Any]
    """A model serializer method without the `info` argument, in `wrap` mode."""

    ModelWrapSerializer: TypeAlias = 'ModelWrapSerializerWithInfo | ModelWrapSerializerWithoutInfo'
    """A model serializer method in `wrap` mode."""

    ModelSerializer: TypeAlias = 'ModelPlainSerializer | ModelWrapSerializer'

    _ModelPlainSerializerT = TypeVar('_ModelPlainSerializerT', bound=ModelPlainSerializer)
    _ModelWrapSerializerT = TypeVar('_ModelWrapSerializerT', bound=ModelWrapSerializer)


@overload
def model_serializer(f: _ModelPlainSerializerT, /) -> _ModelPlainSerializerT: ...


@overload
def model_serializer(
    *, mode: Literal['wrap'], when_used: WhenUsed = 'always', return_type: Any = ...
) -> Callable[[_ModelWrapSerializerT], _ModelWrapSerializerT]: ...


@overload
def model_serializer(
    *,
    mode: Literal['plain'] = ...,
    when_used: WhenUsed = 'always',
    return_type: Any = ...,
) -> Callable[[_ModelPlainSerializerT], _ModelPlainSerializerT]: ...


def model_serializer(
    f: _ModelPlainSerializerT | _ModelWrapSerializerT | None = None,
    /,
    *,
    mode: Literal['plain', 'wrap'] = 'plain',
    when_used: WhenUsed = 'always',
    return_type: Any = PydanticUndefined,
) -> (
    _ModelPlainSerializerT
    | Callable[[_ModelWrapSerializerT], _ModelWrapSerializerT]
    | Callable[[_ModelPlainSerializerT], _ModelPlainSerializerT]
):
    """Decorator that enables custom model serialization.

    See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.

    Args:
        f: The function to be decorated.
        mode: The serialization mode.

            - `'plain'` means the function will be called instead of the default serialization logic
            - `'wrap'` means the function will be called with an argument to optionally call the default
                serialization logic.
        when_used: Determines when this serializer should be used.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.

    Returns:
        The decorator function.
    """

    def dec(f: ModelSerializer) -> _decorators.PydanticDescriptorProxy[Any]:
        dec_info = _decorators.ModelSerializerDecoratorInfo(mode=mode, return_type=return_type, when_used=when_used)
        return _decorators.PydanticDescriptorProxy(f, dec_info)

    if f is None:
        return dec  # pyright: ignore[reportReturnType]
    else:
        return dec(f)  # pyright: ignore[reportReturnType]


AnyType = TypeVar('AnyType')


if TYPE_CHECKING:
    SerializeAsAny = Annotated[AnyType, ...]  # SerializeAsAny[list[str]] will be treated by type checkers as list[str]
    """Force serialization to ignore whatever is defined in the schema and instead ask the object
    itself how it should be serialized.
    In particular, this means that when model subclasses are serialized, fields present in the subclass
    but not in the original schema will be included.
    """
else:

    @dataclasses.dataclass(**_internal_dataclass.slots_true)
    class SerializeAsAny:  # noqa: D101
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, SerializeAsAny()]

        def __get_pydantic_core_schema__(
            self, source_type: Any, handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            schema = handler(source_type)
            schema_to_update = schema
            while schema_to_update['type'] == 'definitions':
                schema_to_update = schema_to_update.copy()
                schema_to_update = schema_to_update['schema']
            schema_to_update['serialization'] = core_schema.wrap_serializer_function_ser_schema(
                lambda x, h: h(x), schema=core_schema.any_schema()
            )
            return schema

        __hash__ = object.__hash__