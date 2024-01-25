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
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import space.core.proto.metadata_pb2
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class DataFile(google.protobuf.message.Message):
    """Information of a data file.
    NEXT_ID: 5
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class Range(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        START_FIELD_NUMBER: builtins.int
        END_FIELD_NUMBER: builtins.int
        start: builtins.int
        """Inclusive."""
        end: builtins.int
        """Exclusive."""
        def __init__(
            self,
            *,
            start: builtins.int = ...,
            end: builtins.int = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["end", b"end", "start", b"start"]) -> None: ...

    PATH_FIELD_NUMBER: builtins.int
    STORAGE_STATISTICS_FIELD_NUMBER: builtins.int
    MANIFEST_FILE_ID_FIELD_NUMBER: builtins.int
    SELECTED_ROWS_FIELD_NUMBER: builtins.int
    path: builtins.str
    """Data file path."""
    @property
    def storage_statistics(self) -> space.core.proto.metadata_pb2.StorageStatistics:
        """Storage statistics of data in the file."""
    manifest_file_id: builtins.int
    """Locally assigned manifest file IDs."""
    @property
    def selected_rows(self) -> global___DataFile.Range:
        """A range of selected rows in the data file.
        Used for partially reading an index file and its records.
        """
    def __init__(
        self,
        *,
        path: builtins.str = ...,
        storage_statistics: space.core.proto.metadata_pb2.StorageStatistics | None = ...,
        manifest_file_id: builtins.int = ...,
        selected_rows: global___DataFile.Range | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["selected_rows", b"selected_rows", "storage_statistics", b"storage_statistics"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["manifest_file_id", b"manifest_file_id", "path", b"path", "selected_rows", b"selected_rows", "storage_statistics", b"storage_statistics"]) -> None: ...

global___DataFile = DataFile

@typing_extensions.final
class FileSet(google.protobuf.message.Message):
    """A set of associated data and manifest files.
    NEXT_ID: 2
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class IndexManifestFilesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.int
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.int = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    INDEX_FILES_FIELD_NUMBER: builtins.int
    INDEX_MANIFEST_FILES_FIELD_NUMBER: builtins.int
    @property
    def index_files(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___DataFile]:
        """Index data files."""
    @property
    def index_manifest_files(self) -> google.protobuf.internal.containers.ScalarMap[builtins.int, builtins.str]:
        """Key is locally assigned manifest IDs by a local operation."""
    def __init__(
        self,
        *,
        index_files: collections.abc.Iterable[global___DataFile] | None = ...,
        index_manifest_files: collections.abc.Mapping[builtins.int, builtins.str] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["index_files", b"index_files", "index_manifest_files", b"index_manifest_files"]) -> None: ...

global___FileSet = FileSet

@typing_extensions.final
class Patch(google.protobuf.message.Message):
    """A patch describing metadata changes to the storage for a data operation.
    NEXT_ID: 5
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ADDITION_FIELD_NUMBER: builtins.int
    DELETION_FIELD_NUMBER: builtins.int
    STORAGE_STATISTICS_UPDATE_FIELD_NUMBER: builtins.int
    CHANGE_LOG_FIELD_NUMBER: builtins.int
    @property
    def addition(self) -> space.core.proto.metadata_pb2.ManifestFiles:
        """Manifest files to add to the storage."""
    @property
    def deletion(self) -> space.core.proto.metadata_pb2.ManifestFiles:
        """Manifest files to remove from the storage."""
    @property
    def storage_statistics_update(self) -> space.core.proto.metadata_pb2.StorageStatistics:
        """The change of the storage statistics."""
    @property
    def change_log(self) -> space.core.proto.metadata_pb2.ChangeLog:
        """The change log describing the changes made by the patch."""
    def __init__(
        self,
        *,
        addition: space.core.proto.metadata_pb2.ManifestFiles | None = ...,
        deletion: space.core.proto.metadata_pb2.ManifestFiles | None = ...,
        storage_statistics_update: space.core.proto.metadata_pb2.StorageStatistics | None = ...,
        change_log: space.core.proto.metadata_pb2.ChangeLog | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["addition", b"addition", "change_log", b"change_log", "deletion", b"deletion", "storage_statistics_update", b"storage_statistics_update"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["addition", b"addition", "change_log", b"change_log", "deletion", b"deletion", "storage_statistics_update", b"storage_statistics_update"]) -> None: ...

global___Patch = Patch
