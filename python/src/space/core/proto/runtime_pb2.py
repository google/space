# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: space/core/proto/runtime.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from space.core.proto import metadata_pb2 as space_dot_core_dot_proto_dot_metadata__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1espace/core/proto/runtime.proto\x12\x0bspace.proto\x1a\x1fspace/core/proto/metadata.proto\"\x94\x01\n\x05Patch\x12\"\n\x1a\x61\x64\x64\x65\x64_index_manifest_files\x18\x01 \x03(\t\x12$\n\x1c\x64\x65leted_index_manifest_files\x18\x02 \x03(\t\x12\x41\n\x19storage_statistics_update\x18\x03 \x01(\x0b\x32\x1e.space.proto.StorageStatistics\"\xc3\x01\n\tJobResult\x12+\n\x05state\x18\x01 \x01(\x0e\x32\x1c.space.proto.JobResult.State\x12\x41\n\x19storage_statistics_update\x18\x02 \x01(\x0b\x32\x1e.space.proto.StorageStatistics\"F\n\x05State\x12\x15\n\x11STATE_UNSPECIFIED\x10\x00\x12\r\n\tSUCCEEDED\x10\x01\x12\n\n\x06\x46\x41ILED\x10\x02\x12\x0b\n\x07SKIPPED\x10\x03\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'space.core.proto.runtime_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PATCH._serialized_start=81
  _PATCH._serialized_end=229
  _JOBRESULT._serialized_start=232
  _JOBRESULT._serialized_end=427
  _JOBRESULT_STATE._serialized_start=357
  _JOBRESULT_STATE._serialized_end=427
# @@protoc_insertion_point(module_scope)
