# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: substrait/type.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14substrait/type.proto\x12\tsubstrait\x1a\x1bgoogle/protobuf/empty.proto\"\xbf\x1e\n\x04Type\x12\'\n\x04\x62ool\x18\x01 \x01(\x0b\x32\x17.substrait.Type.BooleanH\x00\x12 \n\x02i8\x18\x02 \x01(\x0b\x32\x12.substrait.Type.I8H\x00\x12\"\n\x03i16\x18\x03 \x01(\x0b\x32\x13.substrait.Type.I16H\x00\x12\"\n\x03i32\x18\x05 \x01(\x0b\x32\x13.substrait.Type.I32H\x00\x12\"\n\x03i64\x18\x07 \x01(\x0b\x32\x13.substrait.Type.I64H\x00\x12$\n\x04\x66p32\x18\n \x01(\x0b\x32\x14.substrait.Type.FP32H\x00\x12$\n\x04\x66p64\x18\x0b \x01(\x0b\x32\x14.substrait.Type.FP64H\x00\x12(\n\x06string\x18\x0c \x01(\x0b\x32\x16.substrait.Type.StringH\x00\x12(\n\x06\x62inary\x18\r \x01(\x0b\x32\x16.substrait.Type.BinaryH\x00\x12.\n\ttimestamp\x18\x0e \x01(\x0b\x32\x19.substrait.Type.TimestampH\x00\x12$\n\x04\x64\x61te\x18\x10 \x01(\x0b\x32\x14.substrait.Type.DateH\x00\x12$\n\x04time\x18\x11 \x01(\x0b\x32\x14.substrait.Type.TimeH\x00\x12\x35\n\rinterval_year\x18\x13 \x01(\x0b\x32\x1c.substrait.Type.IntervalYearH\x00\x12\x33\n\x0cinterval_day\x18\x14 \x01(\x0b\x32\x1b.substrait.Type.IntervalDayH\x00\x12\x33\n\x0ctimestamp_tz\x18\x1d \x01(\x0b\x32\x1b.substrait.Type.TimestampTZH\x00\x12$\n\x04uuid\x18  \x01(\x0b\x32\x14.substrait.Type.UUIDH\x00\x12/\n\nfixed_char\x18\x15 \x01(\x0b\x32\x19.substrait.Type.FixedCharH\x00\x12*\n\x07varchar\x18\x16 \x01(\x0b\x32\x17.substrait.Type.VarCharH\x00\x12\x33\n\x0c\x66ixed_binary\x18\x17 \x01(\x0b\x32\x1b.substrait.Type.FixedBinaryH\x00\x12*\n\x07\x64\x65\x63imal\x18\x18 \x01(\x0b\x32\x17.substrait.Type.DecimalH\x00\x12(\n\x06struct\x18\x19 \x01(\x0b\x32\x16.substrait.Type.StructH\x00\x12$\n\x04list\x18\x1b \x01(\x0b\x32\x14.substrait.Type.ListH\x00\x12\"\n\x03map\x18\x1c \x01(\x0b\x32\x13.substrait.Type.MapH\x00\x12\x33\n\x0cuser_defined\x18\x1e \x01(\x0b\x32\x1b.substrait.Type.UserDefinedH\x00\x12)\n\x1buser_defined_type_reference\x18\x1f \x01(\rB\x02\x18\x01H\x00\x1a]\n\x07\x42oolean\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aX\n\x02I8\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aY\n\x03I16\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aY\n\x03I32\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aY\n\x03I64\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aZ\n\x04\x46P32\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aZ\n\x04\x46P64\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a\\\n\x06String\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a\\\n\x06\x42inary\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a_\n\tTimestamp\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aZ\n\x04\x44\x61te\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aZ\n\x04Time\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a\x61\n\x0bTimestampTZ\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a\x62\n\x0cIntervalYear\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a\x61\n\x0bIntervalDay\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aZ\n\x04UUID\x12 \n\x18type_variation_reference\x18\x01 \x01(\r\x12\x30\n\x0bnullability\x18\x02 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1ao\n\tFixedChar\x12\x0e\n\x06length\x18\x01 \x01(\x05\x12 \n\x18type_variation_reference\x18\x02 \x01(\r\x12\x30\n\x0bnullability\x18\x03 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1am\n\x07VarChar\x12\x0e\n\x06length\x18\x01 \x01(\x05\x12 \n\x18type_variation_reference\x18\x02 \x01(\r\x12\x30\n\x0bnullability\x18\x03 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1aq\n\x0b\x46ixedBinary\x12\x0e\n\x06length\x18\x01 \x01(\x05\x12 \n\x18type_variation_reference\x18\x02 \x01(\r\x12\x30\n\x0bnullability\x18\x03 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a\x7f\n\x07\x44\x65\x63imal\x12\r\n\x05scale\x18\x01 \x01(\x05\x12\x11\n\tprecision\x18\x02 \x01(\x05\x12 \n\x18type_variation_reference\x18\x03 \x01(\r\x12\x30\n\x0bnullability\x18\x04 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a|\n\x06Struct\x12\x1e\n\x05types\x18\x01 \x03(\x0b\x32\x0f.substrait.Type\x12 \n\x18type_variation_reference\x18\x02 \x01(\r\x12\x30\n\x0bnullability\x18\x03 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1ay\n\x04List\x12\x1d\n\x04type\x18\x01 \x01(\x0b\x32\x0f.substrait.Type\x12 \n\x18type_variation_reference\x18\x02 \x01(\r\x12\x30\n\x0bnullability\x18\x03 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a\x97\x01\n\x03Map\x12\x1c\n\x03key\x18\x01 \x01(\x0b\x32\x0f.substrait.Type\x12\x1e\n\x05value\x18\x02 \x01(\x0b\x32\x0f.substrait.Type\x12 \n\x18type_variation_reference\x18\x03 \x01(\r\x12\x30\n\x0bnullability\x18\x04 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x1a\xad\x01\n\x0bUserDefined\x12\x16\n\x0etype_reference\x18\x01 \x01(\r\x12 \n\x18type_variation_reference\x18\x02 \x01(\r\x12\x30\n\x0bnullability\x18\x03 \x01(\x0e\x32\x1b.substrait.Type.Nullability\x12\x32\n\x0ftype_parameters\x18\x04 \x03(\x0b\x32\x19.substrait.Type.Parameter\x1a\xae\x01\n\tParameter\x12&\n\x04null\x18\x01 \x01(\x0b\x32\x16.google.protobuf.EmptyH\x00\x12$\n\tdata_type\x18\x02 \x01(\x0b\x32\x0f.substrait.TypeH\x00\x12\x11\n\x07\x62oolean\x18\x03 \x01(\x08H\x00\x12\x11\n\x07integer\x18\x04 \x01(\x03H\x00\x12\x0e\n\x04\x65num\x18\x05 \x01(\tH\x00\x12\x10\n\x06string\x18\x06 \x01(\tH\x00\x42\x0b\n\tparameter\"^\n\x0bNullability\x12\x1b\n\x17NULLABILITY_UNSPECIFIED\x10\x00\x12\x18\n\x14NULLABILITY_NULLABLE\x10\x01\x12\x18\n\x14NULLABILITY_REQUIRED\x10\x02\x42\x06\n\x04kind\"D\n\x0bNamedStruct\x12\r\n\x05names\x18\x01 \x03(\t\x12&\n\x06struct\x18\x02 \x01(\x0b\x32\x16.substrait.Type.StructBW\n\x12io.substrait.protoP\x01Z*github.com/substrait-io/substrait-go/proto\xaa\x02\x12Substrait.Protobufb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'substrait.type_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\022io.substrait.protoP\001Z*github.com/substrait-io/substrait-go/proto\252\002\022Substrait.Protobuf'
  _TYPE.fields_by_name['user_defined_type_reference']._options = None
  _TYPE.fields_by_name['user_defined_type_reference']._serialized_options = b'\030\001'
  _TYPE._serialized_start=65
  _TYPE._serialized_end=3968
  _TYPE_BOOLEAN._serialized_start=1141
  _TYPE_BOOLEAN._serialized_end=1234
  _TYPE_I8._serialized_start=1236
  _TYPE_I8._serialized_end=1324
  _TYPE_I16._serialized_start=1326
  _TYPE_I16._serialized_end=1415
  _TYPE_I32._serialized_start=1417
  _TYPE_I32._serialized_end=1506
  _TYPE_I64._serialized_start=1508
  _TYPE_I64._serialized_end=1597
  _TYPE_FP32._serialized_start=1599
  _TYPE_FP32._serialized_end=1689
  _TYPE_FP64._serialized_start=1691
  _TYPE_FP64._serialized_end=1781
  _TYPE_STRING._serialized_start=1783
  _TYPE_STRING._serialized_end=1875
  _TYPE_BINARY._serialized_start=1877
  _TYPE_BINARY._serialized_end=1969
  _TYPE_TIMESTAMP._serialized_start=1971
  _TYPE_TIMESTAMP._serialized_end=2066
  _TYPE_DATE._serialized_start=2068
  _TYPE_DATE._serialized_end=2158
  _TYPE_TIME._serialized_start=2160
  _TYPE_TIME._serialized_end=2250
  _TYPE_TIMESTAMPTZ._serialized_start=2252
  _TYPE_TIMESTAMPTZ._serialized_end=2349
  _TYPE_INTERVALYEAR._serialized_start=2351
  _TYPE_INTERVALYEAR._serialized_end=2449
  _TYPE_INTERVALDAY._serialized_start=2451
  _TYPE_INTERVALDAY._serialized_end=2548
  _TYPE_UUID._serialized_start=2550
  _TYPE_UUID._serialized_end=2640
  _TYPE_FIXEDCHAR._serialized_start=2642
  _TYPE_FIXEDCHAR._serialized_end=2753
  _TYPE_VARCHAR._serialized_start=2755
  _TYPE_VARCHAR._serialized_end=2864
  _TYPE_FIXEDBINARY._serialized_start=2866
  _TYPE_FIXEDBINARY._serialized_end=2979
  _TYPE_DECIMAL._serialized_start=2981
  _TYPE_DECIMAL._serialized_end=3108
  _TYPE_STRUCT._serialized_start=3110
  _TYPE_STRUCT._serialized_end=3234
  _TYPE_LIST._serialized_start=3236
  _TYPE_LIST._serialized_end=3357
  _TYPE_MAP._serialized_start=3360
  _TYPE_MAP._serialized_end=3511
  _TYPE_USERDEFINED._serialized_start=3514
  _TYPE_USERDEFINED._serialized_end=3687
  _TYPE_PARAMETER._serialized_start=3690
  _TYPE_PARAMETER._serialized_end=3864
  _TYPE_NULLABILITY._serialized_start=3866
  _TYPE_NULLABILITY._serialized_end=3960
  _NAMEDSTRUCT._serialized_start=3970
  _NAMEDSTRUCT._serialized_end=4038
# @@protoc_insertion_point(module_scope)
