# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: space/core/proto/metadata.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from substrait import plan_pb2 as substrait_dot_plan__pb2
from substrait import type_pb2 as substrait_dot_type__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1fspace/core/proto/metadata.proto\x12\x0bspace.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x14substrait/plan.proto\x1a\x14substrait/type.proto\"#\n\nEntryPoint\x12\x15\n\rmetadata_file\x18\x01 \x01(\t\"\xe9\x04\n\x0fStorageMetadata\x12/\n\x0b\x63reate_time\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x34\n\x10last_update_time\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12/\n\x04type\x18\x03 \x01(\x0e\x32!.space.proto.StorageMetadata.Type\x12#\n\x06schema\x18\x04 \x01(\x0b\x32\x13.space.proto.Schema\x12\x1b\n\x13\x63urrent_snapshot_id\x18\x05 \x01(\x03\x12>\n\tsnapshots\x18\x06 \x03(\x0b\x32+.space.proto.StorageMetadata.SnapshotsEntry\x12.\n\x0clogical_plan\x18\x07 \x01(\x0b\x32\x18.space.proto.LogicalPlan\x12\x34\n\x04refs\x18\x08 \x03(\x0b\x32&.space.proto.StorageMetadata.RefsEntry\x1aG\n\x0eSnapshotsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.space.proto.Snapshot:\x02\x38\x01\x1aK\n\tRefsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12-\n\x05value\x18\x02 \x01(\x0b\x32\x1e.space.proto.SnapshotReference:\x02\x38\x01\"@\n\x04Type\x12\x14\n\x10TYPE_UNSPECIFIED\x10\x00\x12\x0b\n\x07\x44\x41TASET\x10\x01\x12\x15\n\x11MATERIALIZED_VIEW\x10\x02\"]\n\x06Schema\x12&\n\x06\x66ields\x18\x01 \x01(\x0b\x32\x16.substrait.NamedStruct\x12\x14\n\x0cprimary_keys\x18\x02 \x03(\t\x12\x15\n\rrecord_fields\x18\x03 \x03(\t\"\xee\x01\n\x08Snapshot\x12\x13\n\x0bsnapshot_id\x18\x01 \x01(\x03\x12/\n\x0b\x63reate_time\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x34\n\x0emanifest_files\x18\x03 \x01(\x0b\x32\x1a.space.proto.ManifestFilesH\x00\x12:\n\x12storage_statistics\x18\x04 \x01(\x0b\x32\x1e.space.proto.StorageStatistics\x12\x17\n\x0f\x63hange_log_file\x18\x05 \x01(\tB\x0b\n\tdata_infoJ\x04\x08\x06\x10\x07\"\xb8\x01\n\x11SnapshotReference\x12\x16\n\x0ereference_name\x18\x01 \x01(\t\x12\x13\n\x0bsnapshot_id\x18\x02 \x01(\x03\x12:\n\x04type\x18\x03 \x01(\x0e\x32,.space.proto.SnapshotReference.ReferenceType\":\n\rReferenceType\x12\x14\n\x10TYPE_UNSPECIFIED\x10\x00\x12\x07\n\x03TAG\x10\x01\x12\n\n\x06\x42RANCH\x10\x02\"L\n\rManifestFiles\x12\x1c\n\x14index_manifest_files\x18\x01 \x03(\t\x12\x1d\n\x15record_manifest_files\x18\x02 \x03(\t\"\x8a\x01\n\x11StorageStatistics\x12\x10\n\x08num_rows\x18\x01 \x01(\x03\x12\x1e\n\x16index_compressed_bytes\x18\x02 \x01(\x03\x12 \n\x18index_uncompressed_bytes\x18\x03 \x01(\x03\x12!\n\x19record_uncompressed_bytes\x18\x04 \x01(\x03\"e\n\tChangeLog\x12,\n\x0c\x64\x65leted_rows\x18\x01 \x03(\x0b\x32\x16.space.proto.RowBitmap\x12*\n\nadded_rows\x18\x02 \x03(\x0b\x32\x16.space.proto.RowBitmap\"O\n\tRowBitmap\x12\x0c\n\x04\x66ile\x18\x01 \x01(\t\x12\x10\n\x08\x61ll_rows\x18\x02 \x01(\x08\x12\x18\n\x0eroaring_bitmap\x18\x03 \x01(\x0cH\x00\x42\x08\n\x06\x62itmap\"\x93\x01\n\x0bLogicalPlan\x12%\n\x0clogical_plan\x18\x01 \x01(\x0b\x32\x0f.substrait.Plan\x12\x30\n\x04udfs\x18\x02 \x03(\x0b\x32\".space.proto.LogicalPlan.UdfsEntry\x1a+\n\tUdfsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'space.core.proto.metadata_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _STORAGEMETADATA_SNAPSHOTSENTRY._options = None
  _STORAGEMETADATA_SNAPSHOTSENTRY._serialized_options = b'8\001'
  _STORAGEMETADATA_REFSENTRY._options = None
  _STORAGEMETADATA_REFSENTRY._serialized_options = b'8\001'
  _LOGICALPLAN_UDFSENTRY._options = None
  _LOGICALPLAN_UDFSENTRY._serialized_options = b'8\001'
  _ENTRYPOINT._serialized_start=125
  _ENTRYPOINT._serialized_end=160
  _STORAGEMETADATA._serialized_start=163
  _STORAGEMETADATA._serialized_end=780
  _STORAGEMETADATA_SNAPSHOTSENTRY._serialized_start=566
  _STORAGEMETADATA_SNAPSHOTSENTRY._serialized_end=637
  _STORAGEMETADATA_REFSENTRY._serialized_start=639
  _STORAGEMETADATA_REFSENTRY._serialized_end=714
  _STORAGEMETADATA_TYPE._serialized_start=716
  _STORAGEMETADATA_TYPE._serialized_end=780
  _SCHEMA._serialized_start=782
  _SCHEMA._serialized_end=875
  _SNAPSHOT._serialized_start=878
  _SNAPSHOT._serialized_end=1116
  _SNAPSHOTREFERENCE._serialized_start=1119
  _SNAPSHOTREFERENCE._serialized_end=1303
  _SNAPSHOTREFERENCE_REFERENCETYPE._serialized_start=1245
  _SNAPSHOTREFERENCE_REFERENCETYPE._serialized_end=1303
  _MANIFESTFILES._serialized_start=1305
  _MANIFESTFILES._serialized_end=1381
  _STORAGESTATISTICS._serialized_start=1384
  _STORAGESTATISTICS._serialized_end=1522
  _CHANGELOG._serialized_start=1524
  _CHANGELOG._serialized_end=1625
  _ROWBITMAP._serialized_start=1627
  _ROWBITMAP._serialized_end=1706
  _LOGICALPLAN._serialized_start=1709
  _LOGICALPLAN._serialized_end=1856
  _LOGICALPLAN_UDFSENTRY._serialized_start=1813
  _LOGICALPLAN_UDFSENTRY._serialized_end=1856
# @@protoc_insertion_point(module_scope)
