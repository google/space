# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: substrait/extended_expression.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from substrait import algebra_pb2 as substrait_dot_algebra__pb2
from substrait.extensions import extensions_pb2 as substrait_dot_extensions_dot_extensions__pb2
from substrait import plan_pb2 as substrait_dot_plan__pb2
from substrait import type_pb2 as substrait_dot_type__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n#substrait/extended_expression.proto\x12\tsubstrait\x1a\x17substrait/algebra.proto\x1a%substrait/extensions/extensions.proto\x1a\x14substrait/plan.proto\x1a\x14substrait/type.proto\"\x96\x01\n\x13\x45xpressionReference\x12+\n\nexpression\x18\x01 \x01(\x0b\x32\x15.substrait.ExpressionH\x00\x12/\n\x07measure\x18\x02 \x01(\x0b\x32\x1c.substrait.AggregateFunctionH\x00\x12\x14\n\x0coutput_names\x18\x03 \x03(\tB\x0b\n\texpr_type\"\x87\x03\n\x12\x45xtendedExpression\x12#\n\x07version\x18\x07 \x01(\x0b\x32\x12.substrait.Version\x12@\n\x0e\x65xtension_uris\x18\x01 \x03(\x0b\x32(.substrait.extensions.SimpleExtensionURI\x12\x44\n\nextensions\x18\x02 \x03(\x0b\x32\x30.substrait.extensions.SimpleExtensionDeclaration\x12\x35\n\rreferred_expr\x18\x03 \x03(\x0b\x32\x1e.substrait.ExpressionReference\x12+\n\x0b\x62\x61se_schema\x18\x04 \x01(\x0b\x32\x16.substrait.NamedStruct\x12\x44\n\x13\x61\x64vanced_extensions\x18\x05 \x01(\x0b\x32\'.substrait.extensions.AdvancedExtension\x12\x1a\n\x12\x65xpected_type_urls\x18\x06 \x03(\tBW\n\x12io.substrait.protoP\x01Z*github.com/substrait-io/substrait-go/proto\xaa\x02\x12Substrait.Protobufb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'substrait.extended_expression_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\022io.substrait.protoP\001Z*github.com/substrait-io/substrait-go/proto\252\002\022Substrait.Protobuf'
  _EXPRESSIONREFERENCE._serialized_start=159
  _EXPRESSIONREFERENCE._serialized_end=309
  _EXTENDEDEXPRESSION._serialized_start=312
  _EXTENDEDEXPRESSION._serialized_end=703
# @@protoc_insertion_point(module_scope)
