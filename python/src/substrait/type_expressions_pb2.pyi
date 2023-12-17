"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
SPDX-License-Identifier: Apache-2.0"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import substrait.type_pb2
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class DerivationExpression(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class ExpressionFixedChar(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        LENGTH_FIELD_NUMBER: builtins.int
        VARIATION_POINTER_FIELD_NUMBER: builtins.int
        NULLABILITY_FIELD_NUMBER: builtins.int
        @property
        def length(self) -> global___DerivationExpression: ...
        variation_pointer: builtins.int
        nullability: substrait.type_pb2.Type.Nullability.ValueType
        def __init__(
            self,
            *,
            length: global___DerivationExpression | None = ...,
            variation_pointer: builtins.int = ...,
            nullability: substrait.type_pb2.Type.Nullability.ValueType = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["length", b"length"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["length", b"length", "nullability", b"nullability", "variation_pointer", b"variation_pointer"]) -> None: ...

    @typing_extensions.final
    class ExpressionVarChar(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        LENGTH_FIELD_NUMBER: builtins.int
        VARIATION_POINTER_FIELD_NUMBER: builtins.int
        NULLABILITY_FIELD_NUMBER: builtins.int
        @property
        def length(self) -> global___DerivationExpression: ...
        variation_pointer: builtins.int
        nullability: substrait.type_pb2.Type.Nullability.ValueType
        def __init__(
            self,
            *,
            length: global___DerivationExpression | None = ...,
            variation_pointer: builtins.int = ...,
            nullability: substrait.type_pb2.Type.Nullability.ValueType = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["length", b"length"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["length", b"length", "nullability", b"nullability", "variation_pointer", b"variation_pointer"]) -> None: ...

    @typing_extensions.final
    class ExpressionFixedBinary(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        LENGTH_FIELD_NUMBER: builtins.int
        VARIATION_POINTER_FIELD_NUMBER: builtins.int
        NULLABILITY_FIELD_NUMBER: builtins.int
        @property
        def length(self) -> global___DerivationExpression: ...
        variation_pointer: builtins.int
        nullability: substrait.type_pb2.Type.Nullability.ValueType
        def __init__(
            self,
            *,
            length: global___DerivationExpression | None = ...,
            variation_pointer: builtins.int = ...,
            nullability: substrait.type_pb2.Type.Nullability.ValueType = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["length", b"length"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["length", b"length", "nullability", b"nullability", "variation_pointer", b"variation_pointer"]) -> None: ...

    @typing_extensions.final
    class ExpressionDecimal(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        SCALE_FIELD_NUMBER: builtins.int
        PRECISION_FIELD_NUMBER: builtins.int
        VARIATION_POINTER_FIELD_NUMBER: builtins.int
        NULLABILITY_FIELD_NUMBER: builtins.int
        @property
        def scale(self) -> global___DerivationExpression: ...
        @property
        def precision(self) -> global___DerivationExpression: ...
        variation_pointer: builtins.int
        nullability: substrait.type_pb2.Type.Nullability.ValueType
        def __init__(
            self,
            *,
            scale: global___DerivationExpression | None = ...,
            precision: global___DerivationExpression | None = ...,
            variation_pointer: builtins.int = ...,
            nullability: substrait.type_pb2.Type.Nullability.ValueType = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["precision", b"precision", "scale", b"scale"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["nullability", b"nullability", "precision", b"precision", "scale", b"scale", "variation_pointer", b"variation_pointer"]) -> None: ...

    @typing_extensions.final
    class ExpressionStruct(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        TYPES_FIELD_NUMBER: builtins.int
        VARIATION_POINTER_FIELD_NUMBER: builtins.int
        NULLABILITY_FIELD_NUMBER: builtins.int
        @property
        def types(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___DerivationExpression]: ...
        variation_pointer: builtins.int
        nullability: substrait.type_pb2.Type.Nullability.ValueType
        def __init__(
            self,
            *,
            types: collections.abc.Iterable[global___DerivationExpression] | None = ...,
            variation_pointer: builtins.int = ...,
            nullability: substrait.type_pb2.Type.Nullability.ValueType = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["nullability", b"nullability", "types", b"types", "variation_pointer", b"variation_pointer"]) -> None: ...

    @typing_extensions.final
    class ExpressionNamedStruct(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        NAMES_FIELD_NUMBER: builtins.int
        STRUCT_FIELD_NUMBER: builtins.int
        @property
        def names(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
        @property
        def struct(self) -> global___DerivationExpression.ExpressionStruct: ...
        def __init__(
            self,
            *,
            names: collections.abc.Iterable[builtins.str] | None = ...,
            struct: global___DerivationExpression.ExpressionStruct | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["struct", b"struct"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["names", b"names", "struct", b"struct"]) -> None: ...

    @typing_extensions.final
    class ExpressionList(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        TYPE_FIELD_NUMBER: builtins.int
        VARIATION_POINTER_FIELD_NUMBER: builtins.int
        NULLABILITY_FIELD_NUMBER: builtins.int
        @property
        def type(self) -> global___DerivationExpression: ...
        variation_pointer: builtins.int
        nullability: substrait.type_pb2.Type.Nullability.ValueType
        def __init__(
            self,
            *,
            type: global___DerivationExpression | None = ...,
            variation_pointer: builtins.int = ...,
            nullability: substrait.type_pb2.Type.Nullability.ValueType = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["type", b"type"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["nullability", b"nullability", "type", b"type", "variation_pointer", b"variation_pointer"]) -> None: ...

    @typing_extensions.final
    class ExpressionMap(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        VARIATION_POINTER_FIELD_NUMBER: builtins.int
        NULLABILITY_FIELD_NUMBER: builtins.int
        @property
        def key(self) -> global___DerivationExpression: ...
        @property
        def value(self) -> global___DerivationExpression: ...
        variation_pointer: builtins.int
        nullability: substrait.type_pb2.Type.Nullability.ValueType
        def __init__(
            self,
            *,
            key: global___DerivationExpression | None = ...,
            value: global___DerivationExpression | None = ...,
            variation_pointer: builtins.int = ...,
            nullability: substrait.type_pb2.Type.Nullability.ValueType = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "nullability", b"nullability", "value", b"value", "variation_pointer", b"variation_pointer"]) -> None: ...

    @typing_extensions.final
    class ExpressionUserDefined(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        TYPE_POINTER_FIELD_NUMBER: builtins.int
        VARIATION_POINTER_FIELD_NUMBER: builtins.int
        NULLABILITY_FIELD_NUMBER: builtins.int
        type_pointer: builtins.int
        variation_pointer: builtins.int
        nullability: substrait.type_pb2.Type.Nullability.ValueType
        def __init__(
            self,
            *,
            type_pointer: builtins.int = ...,
            variation_pointer: builtins.int = ...,
            nullability: substrait.type_pb2.Type.Nullability.ValueType = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["nullability", b"nullability", "type_pointer", b"type_pointer", "variation_pointer", b"variation_pointer"]) -> None: ...

    @typing_extensions.final
    class IfElse(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        IF_CONDITION_FIELD_NUMBER: builtins.int
        IF_RETURN_FIELD_NUMBER: builtins.int
        ELSE_RETURN_FIELD_NUMBER: builtins.int
        @property
        def if_condition(self) -> global___DerivationExpression: ...
        @property
        def if_return(self) -> global___DerivationExpression: ...
        @property
        def else_return(self) -> global___DerivationExpression: ...
        def __init__(
            self,
            *,
            if_condition: global___DerivationExpression | None = ...,
            if_return: global___DerivationExpression | None = ...,
            else_return: global___DerivationExpression | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["else_return", b"else_return", "if_condition", b"if_condition", "if_return", b"if_return"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["else_return", b"else_return", "if_condition", b"if_condition", "if_return", b"if_return"]) -> None: ...

    @typing_extensions.final
    class UnaryOp(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        class _UnaryOpType:
            ValueType = typing.NewType("ValueType", builtins.int)
            V: typing_extensions.TypeAlias = ValueType

        class _UnaryOpTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[DerivationExpression.UnaryOp._UnaryOpType.ValueType], builtins.type):
            DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
            UNARY_OP_TYPE_UNSPECIFIED: DerivationExpression.UnaryOp._UnaryOpType.ValueType  # 0
            UNARY_OP_TYPE_BOOLEAN_NOT: DerivationExpression.UnaryOp._UnaryOpType.ValueType  # 1

        class UnaryOpType(_UnaryOpType, metaclass=_UnaryOpTypeEnumTypeWrapper): ...
        UNARY_OP_TYPE_UNSPECIFIED: DerivationExpression.UnaryOp.UnaryOpType.ValueType  # 0
        UNARY_OP_TYPE_BOOLEAN_NOT: DerivationExpression.UnaryOp.UnaryOpType.ValueType  # 1

        OP_TYPE_FIELD_NUMBER: builtins.int
        ARG_FIELD_NUMBER: builtins.int
        op_type: global___DerivationExpression.UnaryOp.UnaryOpType.ValueType
        @property
        def arg(self) -> global___DerivationExpression: ...
        def __init__(
            self,
            *,
            op_type: global___DerivationExpression.UnaryOp.UnaryOpType.ValueType = ...,
            arg: global___DerivationExpression | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["arg", b"arg"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["arg", b"arg", "op_type", b"op_type"]) -> None: ...

    @typing_extensions.final
    class BinaryOp(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        class _BinaryOpType:
            ValueType = typing.NewType("ValueType", builtins.int)
            V: typing_extensions.TypeAlias = ValueType

        class _BinaryOpTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[DerivationExpression.BinaryOp._BinaryOpType.ValueType], builtins.type):
            DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
            BINARY_OP_TYPE_UNSPECIFIED: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 0
            BINARY_OP_TYPE_PLUS: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 1
            BINARY_OP_TYPE_MINUS: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 2
            BINARY_OP_TYPE_MULTIPLY: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 3
            BINARY_OP_TYPE_DIVIDE: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 4
            BINARY_OP_TYPE_MIN: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 5
            BINARY_OP_TYPE_MAX: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 6
            BINARY_OP_TYPE_GREATER_THAN: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 7
            BINARY_OP_TYPE_LESS_THAN: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 8
            BINARY_OP_TYPE_AND: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 9
            BINARY_OP_TYPE_OR: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 10
            BINARY_OP_TYPE_EQUALS: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 11
            BINARY_OP_TYPE_COVERS: DerivationExpression.BinaryOp._BinaryOpType.ValueType  # 12

        class BinaryOpType(_BinaryOpType, metaclass=_BinaryOpTypeEnumTypeWrapper): ...
        BINARY_OP_TYPE_UNSPECIFIED: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 0
        BINARY_OP_TYPE_PLUS: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 1
        BINARY_OP_TYPE_MINUS: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 2
        BINARY_OP_TYPE_MULTIPLY: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 3
        BINARY_OP_TYPE_DIVIDE: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 4
        BINARY_OP_TYPE_MIN: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 5
        BINARY_OP_TYPE_MAX: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 6
        BINARY_OP_TYPE_GREATER_THAN: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 7
        BINARY_OP_TYPE_LESS_THAN: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 8
        BINARY_OP_TYPE_AND: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 9
        BINARY_OP_TYPE_OR: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 10
        BINARY_OP_TYPE_EQUALS: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 11
        BINARY_OP_TYPE_COVERS: DerivationExpression.BinaryOp.BinaryOpType.ValueType  # 12

        OP_TYPE_FIELD_NUMBER: builtins.int
        ARG1_FIELD_NUMBER: builtins.int
        ARG2_FIELD_NUMBER: builtins.int
        op_type: global___DerivationExpression.BinaryOp.BinaryOpType.ValueType
        @property
        def arg1(self) -> global___DerivationExpression: ...
        @property
        def arg2(self) -> global___DerivationExpression: ...
        def __init__(
            self,
            *,
            op_type: global___DerivationExpression.BinaryOp.BinaryOpType.ValueType = ...,
            arg1: global___DerivationExpression | None = ...,
            arg2: global___DerivationExpression | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["arg1", b"arg1", "arg2", b"arg2"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["arg1", b"arg1", "arg2", b"arg2", "op_type", b"op_type"]) -> None: ...

    @typing_extensions.final
    class ReturnProgram(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        @typing_extensions.final
        class Assignment(google.protobuf.message.Message):
            DESCRIPTOR: google.protobuf.descriptor.Descriptor

            NAME_FIELD_NUMBER: builtins.int
            EXPRESSION_FIELD_NUMBER: builtins.int
            name: builtins.str
            @property
            def expression(self) -> global___DerivationExpression: ...
            def __init__(
                self,
                *,
                name: builtins.str = ...,
                expression: global___DerivationExpression | None = ...,
            ) -> None: ...
            def HasField(self, field_name: typing_extensions.Literal["expression", b"expression"]) -> builtins.bool: ...
            def ClearField(self, field_name: typing_extensions.Literal["expression", b"expression", "name", b"name"]) -> None: ...

        ASSIGNMENTS_FIELD_NUMBER: builtins.int
        FINAL_EXPRESSION_FIELD_NUMBER: builtins.int
        @property
        def assignments(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___DerivationExpression.ReturnProgram.Assignment]: ...
        @property
        def final_expression(self) -> global___DerivationExpression: ...
        def __init__(
            self,
            *,
            assignments: collections.abc.Iterable[global___DerivationExpression.ReturnProgram.Assignment] | None = ...,
            final_expression: global___DerivationExpression | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["final_expression", b"final_expression"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["assignments", b"assignments", "final_expression", b"final_expression"]) -> None: ...

    BOOL_FIELD_NUMBER: builtins.int
    I8_FIELD_NUMBER: builtins.int
    I16_FIELD_NUMBER: builtins.int
    I32_FIELD_NUMBER: builtins.int
    I64_FIELD_NUMBER: builtins.int
    FP32_FIELD_NUMBER: builtins.int
    FP64_FIELD_NUMBER: builtins.int
    STRING_FIELD_NUMBER: builtins.int
    BINARY_FIELD_NUMBER: builtins.int
    TIMESTAMP_FIELD_NUMBER: builtins.int
    DATE_FIELD_NUMBER: builtins.int
    TIME_FIELD_NUMBER: builtins.int
    INTERVAL_YEAR_FIELD_NUMBER: builtins.int
    INTERVAL_DAY_FIELD_NUMBER: builtins.int
    TIMESTAMP_TZ_FIELD_NUMBER: builtins.int
    UUID_FIELD_NUMBER: builtins.int
    FIXED_CHAR_FIELD_NUMBER: builtins.int
    VARCHAR_FIELD_NUMBER: builtins.int
    FIXED_BINARY_FIELD_NUMBER: builtins.int
    DECIMAL_FIELD_NUMBER: builtins.int
    STRUCT_FIELD_NUMBER: builtins.int
    LIST_FIELD_NUMBER: builtins.int
    MAP_FIELD_NUMBER: builtins.int
    USER_DEFINED_FIELD_NUMBER: builtins.int
    USER_DEFINED_POINTER_FIELD_NUMBER: builtins.int
    TYPE_PARAMETER_NAME_FIELD_NUMBER: builtins.int
    INTEGER_PARAMETER_NAME_FIELD_NUMBER: builtins.int
    INTEGER_LITERAL_FIELD_NUMBER: builtins.int
    UNARY_OP_FIELD_NUMBER: builtins.int
    BINARY_OP_FIELD_NUMBER: builtins.int
    IF_ELSE_FIELD_NUMBER: builtins.int
    RETURN_PROGRAM_FIELD_NUMBER: builtins.int
    @property
    def bool(self) -> substrait.type_pb2.Type.Boolean: ...
    @property
    def i8(self) -> substrait.type_pb2.Type.I8: ...
    @property
    def i16(self) -> substrait.type_pb2.Type.I16: ...
    @property
    def i32(self) -> substrait.type_pb2.Type.I32: ...
    @property
    def i64(self) -> substrait.type_pb2.Type.I64: ...
    @property
    def fp32(self) -> substrait.type_pb2.Type.FP32: ...
    @property
    def fp64(self) -> substrait.type_pb2.Type.FP64: ...
    @property
    def string(self) -> substrait.type_pb2.Type.String: ...
    @property
    def binary(self) -> substrait.type_pb2.Type.Binary: ...
    @property
    def timestamp(self) -> substrait.type_pb2.Type.Timestamp: ...
    @property
    def date(self) -> substrait.type_pb2.Type.Date: ...
    @property
    def time(self) -> substrait.type_pb2.Type.Time: ...
    @property
    def interval_year(self) -> substrait.type_pb2.Type.IntervalYear: ...
    @property
    def interval_day(self) -> substrait.type_pb2.Type.IntervalDay: ...
    @property
    def timestamp_tz(self) -> substrait.type_pb2.Type.TimestampTZ: ...
    @property
    def uuid(self) -> substrait.type_pb2.Type.UUID: ...
    @property
    def fixed_char(self) -> global___DerivationExpression.ExpressionFixedChar: ...
    @property
    def varchar(self) -> global___DerivationExpression.ExpressionVarChar: ...
    @property
    def fixed_binary(self) -> global___DerivationExpression.ExpressionFixedBinary: ...
    @property
    def decimal(self) -> global___DerivationExpression.ExpressionDecimal: ...
    @property
    def struct(self) -> global___DerivationExpression.ExpressionStruct: ...
    @property
    def list(self) -> global___DerivationExpression.ExpressionList: ...
    @property
    def map(self) -> global___DerivationExpression.ExpressionMap: ...
    @property
    def user_defined(self) -> global___DerivationExpression.ExpressionUserDefined: ...
    user_defined_pointer: builtins.int
    """Deprecated in favor of user_defined, which allows nullability and
    variations to be specified. If user_defined_pointer is encountered,
    treat it as being non-nullable and having the default variation.
    """
    type_parameter_name: builtins.str
    integer_parameter_name: builtins.str
    integer_literal: builtins.int
    @property
    def unary_op(self) -> global___DerivationExpression.UnaryOp: ...
    @property
    def binary_op(self) -> global___DerivationExpression.BinaryOp: ...
    @property
    def if_else(self) -> global___DerivationExpression.IfElse: ...
    @property
    def return_program(self) -> global___DerivationExpression.ReturnProgram: ...
    def __init__(
        self,
        *,
        bool: substrait.type_pb2.Type.Boolean | None = ...,
        i8: substrait.type_pb2.Type.I8 | None = ...,
        i16: substrait.type_pb2.Type.I16 | None = ...,
        i32: substrait.type_pb2.Type.I32 | None = ...,
        i64: substrait.type_pb2.Type.I64 | None = ...,
        fp32: substrait.type_pb2.Type.FP32 | None = ...,
        fp64: substrait.type_pb2.Type.FP64 | None = ...,
        string: substrait.type_pb2.Type.String | None = ...,
        binary: substrait.type_pb2.Type.Binary | None = ...,
        timestamp: substrait.type_pb2.Type.Timestamp | None = ...,
        date: substrait.type_pb2.Type.Date | None = ...,
        time: substrait.type_pb2.Type.Time | None = ...,
        interval_year: substrait.type_pb2.Type.IntervalYear | None = ...,
        interval_day: substrait.type_pb2.Type.IntervalDay | None = ...,
        timestamp_tz: substrait.type_pb2.Type.TimestampTZ | None = ...,
        uuid: substrait.type_pb2.Type.UUID | None = ...,
        fixed_char: global___DerivationExpression.ExpressionFixedChar | None = ...,
        varchar: global___DerivationExpression.ExpressionVarChar | None = ...,
        fixed_binary: global___DerivationExpression.ExpressionFixedBinary | None = ...,
        decimal: global___DerivationExpression.ExpressionDecimal | None = ...,
        struct: global___DerivationExpression.ExpressionStruct | None = ...,
        list: global___DerivationExpression.ExpressionList | None = ...,
        map: global___DerivationExpression.ExpressionMap | None = ...,
        user_defined: global___DerivationExpression.ExpressionUserDefined | None = ...,
        user_defined_pointer: builtins.int = ...,
        type_parameter_name: builtins.str = ...,
        integer_parameter_name: builtins.str = ...,
        integer_literal: builtins.int = ...,
        unary_op: global___DerivationExpression.UnaryOp | None = ...,
        binary_op: global___DerivationExpression.BinaryOp | None = ...,
        if_else: global___DerivationExpression.IfElse | None = ...,
        return_program: global___DerivationExpression.ReturnProgram | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["binary", b"binary", "binary_op", b"binary_op", "bool", b"bool", "date", b"date", "decimal", b"decimal", "fixed_binary", b"fixed_binary", "fixed_char", b"fixed_char", "fp32", b"fp32", "fp64", b"fp64", "i16", b"i16", "i32", b"i32", "i64", b"i64", "i8", b"i8", "if_else", b"if_else", "integer_literal", b"integer_literal", "integer_parameter_name", b"integer_parameter_name", "interval_day", b"interval_day", "interval_year", b"interval_year", "kind", b"kind", "list", b"list", "map", b"map", "return_program", b"return_program", "string", b"string", "struct", b"struct", "time", b"time", "timestamp", b"timestamp", "timestamp_tz", b"timestamp_tz", "type_parameter_name", b"type_parameter_name", "unary_op", b"unary_op", "user_defined", b"user_defined", "user_defined_pointer", b"user_defined_pointer", "uuid", b"uuid", "varchar", b"varchar"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["binary", b"binary", "binary_op", b"binary_op", "bool", b"bool", "date", b"date", "decimal", b"decimal", "fixed_binary", b"fixed_binary", "fixed_char", b"fixed_char", "fp32", b"fp32", "fp64", b"fp64", "i16", b"i16", "i32", b"i32", "i64", b"i64", "i8", b"i8", "if_else", b"if_else", "integer_literal", b"integer_literal", "integer_parameter_name", b"integer_parameter_name", "interval_day", b"interval_day", "interval_year", b"interval_year", "kind", b"kind", "list", b"list", "map", b"map", "return_program", b"return_program", "string", b"string", "struct", b"struct", "time", b"time", "timestamp", b"timestamp", "timestamp_tz", b"timestamp_tz", "type_parameter_name", b"type_parameter_name", "unary_op", b"unary_op", "user_defined", b"user_defined", "user_defined_pointer", b"user_defined_pointer", "uuid", b"uuid", "varchar", b"varchar"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["kind", b"kind"]) -> typing_extensions.Literal["bool", "i8", "i16", "i32", "i64", "fp32", "fp64", "string", "binary", "timestamp", "date", "time", "interval_year", "interval_day", "timestamp_tz", "uuid", "fixed_char", "varchar", "fixed_binary", "decimal", "struct", "list", "map", "user_defined", "user_defined_pointer", "type_parameter_name", "integer_parameter_name", "integer_literal", "unary_op", "binary_op", "if_else", "return_program"] | None: ...

global___DerivationExpression = DerivationExpression
