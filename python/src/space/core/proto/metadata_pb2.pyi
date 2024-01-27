"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Proto messages used by Space metadata persistence.
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import google.protobuf.timestamp_pb2
import substrait.plan_pb2
import substrait.type_pb2
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class EntryPoint(google.protobuf.message.Message):
    """Record the current storage metadata path in a static local file.
    A mutation to storage generates a new metadata file. The current metadata
    file path is either persisted in the entry point file, or an external
    catalog (not implemented yet).
    NEXT_ID: 2
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    METADATA_FILE_FIELD_NUMBER: builtins.int
    metadata_file: builtins.str
    """File path of the current storage metadata file."""
    def __init__(
        self,
        *,
        metadata_file: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["metadata_file", b"metadata_file"]) -> None: ...

global___EntryPoint = EntryPoint

@typing_extensions.final
class StorageMetadata(google.protobuf.message.Message):
    """Metadata persisting the current status of a storage, including logical
    metadata such as schema, and physical metadata persisted as a history of
    snapshots
    NEXT_ID: 9
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _Type:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _TypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[StorageMetadata._Type.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        TYPE_UNSPECIFIED: StorageMetadata._Type.ValueType  # 0
        DATASET: StorageMetadata._Type.ValueType  # 1
        """Dataset type supports fully managed storage features."""
        MATERIALIZED_VIEW: StorageMetadata._Type.ValueType  # 2
        """Materialized view type supports synchronizing changes from sources."""

    class Type(_Type, metaclass=_TypeEnumTypeWrapper):
        """The storage type."""

    TYPE_UNSPECIFIED: StorageMetadata.Type.ValueType  # 0
    DATASET: StorageMetadata.Type.ValueType  # 1
    """Dataset type supports fully managed storage features."""
    MATERIALIZED_VIEW: StorageMetadata.Type.ValueType  # 2
    """Materialized view type supports synchronizing changes from sources."""

    @typing_extensions.final
    class SnapshotsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.int
        @property
        def value(self) -> global___Snapshot: ...
        def __init__(
            self,
            *,
            key: builtins.int = ...,
            value: global___Snapshot | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    @typing_extensions.final
    class RefsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        @property
        def value(self) -> global___SnapshotReference: ...
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: global___SnapshotReference | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    CREATE_TIME_FIELD_NUMBER: builtins.int
    LAST_UPDATE_TIME_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    SCHEMA_FIELD_NUMBER: builtins.int
    CURRENT_SNAPSHOT_ID_FIELD_NUMBER: builtins.int
    SNAPSHOTS_FIELD_NUMBER: builtins.int
    LOGICAL_PLAN_FIELD_NUMBER: builtins.int
    REFS_FIELD_NUMBER: builtins.int
    @property
    def create_time(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Create time of the storage."""
    @property
    def last_update_time(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Last update time of the storage."""
    type: global___StorageMetadata.Type.ValueType
    @property
    def schema(self) -> global___Schema:
        """The storage schema."""
    current_snapshot_id: builtins.int
    """The current snapshot ID for the main branch."""
    @property
    def snapshots(self) -> google.protobuf.internal.containers.MessageMap[builtins.int, global___Snapshot]:
        """All alive snapshots with snapshot ID as key."""
    @property
    def logical_plan(self) -> global___LogicalPlan:
        """Store the logical plan for materialized views."""
    @property
    def refs(self) -> google.protobuf.internal.containers.MessageMap[builtins.str, global___SnapshotReference]:
        """All alive refs, with reference name as key. Reference name can be a tag 
        or a branch name.
        """
    def __init__(
        self,
        *,
        create_time: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        last_update_time: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        type: global___StorageMetadata.Type.ValueType = ...,
        schema: global___Schema | None = ...,
        current_snapshot_id: builtins.int = ...,
        snapshots: collections.abc.Mapping[builtins.int, global___Snapshot] | None = ...,
        logical_plan: global___LogicalPlan | None = ...,
        refs: collections.abc.Mapping[builtins.str, global___SnapshotReference] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["create_time", b"create_time", "last_update_time", b"last_update_time", "logical_plan", b"logical_plan", "schema", b"schema"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["create_time", b"create_time", "current_snapshot_id", b"current_snapshot_id", "last_update_time", b"last_update_time", "logical_plan", b"logical_plan", "refs", b"refs", "schema", b"schema", "snapshots", b"snapshots", "type", b"type"]) -> None: ...

global___StorageMetadata = StorageMetadata

@typing_extensions.final
class Schema(google.protobuf.message.Message):
    """The storage logical schema where user provided types are persisted instead
    of their physical storage format.
    NEXT_ID: 4
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FIELDS_FIELD_NUMBER: builtins.int
    PRIMARY_KEYS_FIELD_NUMBER: builtins.int
    RECORD_FIELDS_FIELD_NUMBER: builtins.int
    @property
    def fields(self) -> substrait.type_pb2.NamedStruct:
        """Fields persisted as Substrait named struct."""
    @property
    def primary_keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Primary key field names. Required but primary keys are un-enforced."""
    @property
    def record_fields(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Names of record fields that are stored in row formats (ArrayRecord)."""
    def __init__(
        self,
        *,
        fields: substrait.type_pb2.NamedStruct | None = ...,
        primary_keys: collections.abc.Iterable[builtins.str] | None = ...,
        record_fields: collections.abc.Iterable[builtins.str] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["fields", b"fields"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["fields", b"fields", "primary_keys", b"primary_keys", "record_fields", b"record_fields"]) -> None: ...

global___Schema = Schema

@typing_extensions.final
class Snapshot(google.protobuf.message.Message):
    """Storage snapshot persisting physical metadata such as manifest file paths.
    It is used for obtaining all alive data file paths for a given snapshot.
    NEXT_ID: 7
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SNAPSHOT_ID_FIELD_NUMBER: builtins.int
    CREATE_TIME_FIELD_NUMBER: builtins.int
    MANIFEST_FILES_FIELD_NUMBER: builtins.int
    STORAGE_STATISTICS_FIELD_NUMBER: builtins.int
    CHANGE_LOG_FILE_FIELD_NUMBER: builtins.int
    PARENT_SNAPSHOT_ID_FIELD_NUMBER: builtins.int
    snapshot_id: builtins.int
    """The snapshot ID."""
    @property
    def create_time(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """The create time of the snapshot."""
    @property
    def manifest_files(self) -> global___ManifestFiles:
        """Manifest file information embedded in Snapshot. Preferred option when
        the number of manifest files are small.
        """
    @property
    def storage_statistics(self) -> global___StorageStatistics:
        """Statistics of all data in the storage."""
    change_log_file: builtins.str
    """File path of the change log of the snapshot."""
    parent_snapshot_id: builtins.int
    """The snapshot ID of the parent snapshot."""
    def __init__(
        self,
        *,
        snapshot_id: builtins.int = ...,
        create_time: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        manifest_files: global___ManifestFiles | None = ...,
        storage_statistics: global___StorageStatistics | None = ...,
        change_log_file: builtins.str = ...,
        parent_snapshot_id: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_parent_snapshot_id", b"_parent_snapshot_id", "create_time", b"create_time", "data_info", b"data_info", "manifest_files", b"manifest_files", "parent_snapshot_id", b"parent_snapshot_id", "storage_statistics", b"storage_statistics"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_parent_snapshot_id", b"_parent_snapshot_id", "change_log_file", b"change_log_file", "create_time", b"create_time", "data_info", b"data_info", "manifest_files", b"manifest_files", "parent_snapshot_id", b"parent_snapshot_id", "snapshot_id", b"snapshot_id", "storage_statistics", b"storage_statistics"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_parent_snapshot_id", b"_parent_snapshot_id"]) -> typing_extensions.Literal["parent_snapshot_id"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["data_info", b"data_info"]) -> typing_extensions.Literal["manifest_files"] | None: ...

global___Snapshot = Snapshot

@typing_extensions.final
class SnapshotReference(google.protobuf.message.Message):
    """Reference to a snapshot.
    NEXT_ID: 4
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _ReferenceType:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _ReferenceTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[SnapshotReference._ReferenceType.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        TYPE_UNSPECIFIED: SnapshotReference._ReferenceType.ValueType  # 0
        TAG: SnapshotReference._ReferenceType.ValueType  # 1
        """Reference of a specific snapshot within the storage history."""
        BRANCH: SnapshotReference._ReferenceType.ValueType  # 2
        """Reference of the current snapshot of a branch."""

    class ReferenceType(_ReferenceType, metaclass=_ReferenceTypeEnumTypeWrapper): ...
    TYPE_UNSPECIFIED: SnapshotReference.ReferenceType.ValueType  # 0
    TAG: SnapshotReference.ReferenceType.ValueType  # 1
    """Reference of a specific snapshot within the storage history."""
    BRANCH: SnapshotReference.ReferenceType.ValueType  # 2
    """Reference of the current snapshot of a branch."""

    REFERENCE_NAME_FIELD_NUMBER: builtins.int
    SNAPSHOT_ID_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    reference_name: builtins.str
    """Name for the reference."""
    snapshot_id: builtins.int
    """The snapshot ID."""
    type: global___SnapshotReference.ReferenceType.ValueType
    def __init__(
        self,
        *,
        reference_name: builtins.str = ...,
        snapshot_id: builtins.int = ...,
        type: global___SnapshotReference.ReferenceType.ValueType = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["reference_name", b"reference_name", "snapshot_id", b"snapshot_id", "type", b"type"]) -> None: ...

global___SnapshotReference = SnapshotReference

@typing_extensions.final
class ManifestFiles(google.protobuf.message.Message):
    """Stores information of manifest files.
    NEXT_ID: 3
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INDEX_MANIFEST_FILES_FIELD_NUMBER: builtins.int
    RECORD_MANIFEST_FILES_FIELD_NUMBER: builtins.int
    @property
    def index_manifest_files(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Manifest for index files."""
    @property
    def record_manifest_files(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Manifest for record files."""
    def __init__(
        self,
        *,
        index_manifest_files: collections.abc.Iterable[builtins.str] | None = ...,
        record_manifest_files: collections.abc.Iterable[builtins.str] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["index_manifest_files", b"index_manifest_files", "record_manifest_files", b"record_manifest_files"]) -> None: ...

global___ManifestFiles = ManifestFiles

@typing_extensions.final
class StorageStatistics(google.protobuf.message.Message):
    """Statistics of storage data.
    NEXT_ID: 5
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NUM_ROWS_FIELD_NUMBER: builtins.int
    INDEX_COMPRESSED_BYTES_FIELD_NUMBER: builtins.int
    INDEX_UNCOMPRESSED_BYTES_FIELD_NUMBER: builtins.int
    RECORD_UNCOMPRESSED_BYTES_FIELD_NUMBER: builtins.int
    num_rows: builtins.int
    """Number of rows."""
    index_compressed_bytes: builtins.int
    """Compressed bytes of index data."""
    index_uncompressed_bytes: builtins.int
    """Uncompressed bytes of index data."""
    record_uncompressed_bytes: builtins.int
    """Uncompressed bytes of record data."""
    def __init__(
        self,
        *,
        num_rows: builtins.int = ...,
        index_compressed_bytes: builtins.int = ...,
        index_uncompressed_bytes: builtins.int = ...,
        record_uncompressed_bytes: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["index_compressed_bytes", b"index_compressed_bytes", "index_uncompressed_bytes", b"index_uncompressed_bytes", "num_rows", b"num_rows", "record_uncompressed_bytes", b"record_uncompressed_bytes"]) -> None: ...

global___StorageStatistics = StorageStatistics

@typing_extensions.final
class ChangeLog(google.protobuf.message.Message):
    """Change log stores changes made by a snapshot.
    NEXT_ID: 3
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DELETED_ROWS_FIELD_NUMBER: builtins.int
    ADDED_ROWS_FIELD_NUMBER: builtins.int
    @property
    def deleted_rows(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RowBitmap]:
        """Rows deleted in this snapshot."""
    @property
    def added_rows(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RowBitmap]:
        """New rows added in this snapshot."""
    def __init__(
        self,
        *,
        deleted_rows: collections.abc.Iterable[global___RowBitmap] | None = ...,
        added_rows: collections.abc.Iterable[global___RowBitmap] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["added_rows", b"added_rows", "deleted_rows", b"deleted_rows"]) -> None: ...

global___ChangeLog = ChangeLog

@typing_extensions.final
class RowBitmap(google.protobuf.message.Message):
    """Mark rows in a file by bitmap.
    NEXT_ID: 5
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FILE_FIELD_NUMBER: builtins.int
    ALL_ROWS_FIELD_NUMBER: builtins.int
    ROARING_BITMAP_FIELD_NUMBER: builtins.int
    NUM_ROWS_FIELD_NUMBER: builtins.int
    file: builtins.str
    """File path that the bit map applies to."""
    all_rows: builtins.bool
    """All rows are selected. Bitmap is empty in this case."""
    roaring_bitmap: builtins.bytes
    """Roaring bitmap."""
    num_rows: builtins.int
    """Total number of rows in the file."""
    def __init__(
        self,
        *,
        file: builtins.str = ...,
        all_rows: builtins.bool = ...,
        roaring_bitmap: builtins.bytes = ...,
        num_rows: builtins.int = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["bitmap", b"bitmap", "roaring_bitmap", b"roaring_bitmap"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["all_rows", b"all_rows", "bitmap", b"bitmap", "file", b"file", "num_rows", b"num_rows", "roaring_bitmap", b"roaring_bitmap"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["bitmap", b"bitmap"]) -> typing_extensions.Literal["roaring_bitmap"] | None: ...

global___RowBitmap = RowBitmap

@typing_extensions.final
class LogicalPlan(google.protobuf.message.Message):
    """Store the logical plan of a transform.
    NEXT_ID: 3
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class UdfsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    LOGICAL_PLAN_FIELD_NUMBER: builtins.int
    UDFS_FIELD_NUMBER: builtins.int
    @property
    def logical_plan(self) -> substrait.plan_pb2.Plan:
        """Stores the logical plan."""
    @property
    def udfs(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]:
        """Registry of user defined functions.
        Key is UDF name; value is pickle file path.
        """
    def __init__(
        self,
        *,
        logical_plan: substrait.plan_pb2.Plan | None = ...,
        udfs: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["logical_plan", b"logical_plan"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["logical_plan", b"logical_plan", "udfs", b"udfs"]) -> None: ...

global___LogicalPlan = LogicalPlan

@typing_extensions.final
class FileType(google.protobuf.message.Message):
    """A field type representing a file.
    NEXT_ID: 2
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DIRECTORY_FIELD_NUMBER: builtins.int
    directory: builtins.str
    """The common directory of all files stored as the field.
    Used as the path prefix when read or write files.
    """
    def __init__(
        self,
        *,
        directory: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["directory", b"directory"]) -> None: ...

global___FileType = FileType
