import contextlib
import logging
import os
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    BadRequestException,
    NotFoundException,
    ServiceUnavailableException,
)

logger = logging.getLogger("app.modules.documents.storage")

STORAGE_KEY_PATTERN = re.compile(
    r"^users/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"/documents/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(\.(pdf|docx|txt)|/assets/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(png|jpg|jpeg|webp|gif))$"
)
STREAM_CHUNK_SIZE = 64 * 1024


def build_storage_key(user_id: uuid.UUID, document_id: uuid.UUID, extension: str) -> str:
    ext = extension.lower().lstrip(".")
    key = f"users/{user_id}/documents/{document_id}.{ext}"
    validate_storage_key(key)
    return key


def build_asset_storage_key(
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    asset_id: uuid.UUID,
    extension: str,
) -> str:
    ext = extension.lower().lstrip(".")
    key = f"users/{user_id}/documents/{document_id}/assets/{asset_id}.{ext}"
    validate_storage_key(key)
    return key


def validate_storage_key(storage_key: str) -> None:
    if not STORAGE_KEY_PATTERN.fullmatch(storage_key):
        raise BadRequestException(
            "Invalid internal storage key format",
            code="INVALID_STORAGE_KEY",
        )


class BinaryReadableStream(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...

    def seek(self, offset: int, whence: int = 0, /) -> int: ...


class StorageProvider(Protocol):
    def save(self, storage_key: str, stream: BinaryReadableStream, content_type: str) -> None: ...

    def open_stream(self, storage_key: str) -> Iterator[bytes]: ...

    def delete(self, storage_key: str) -> None: ...

    def exists(self, storage_key: str) -> bool: ...


class LocalStorageProvider:
    """Stores uploaded documents in a private local directory outside public static assets."""

    def __init__(self, base_directory: str | Path) -> None:
        self.base_dir = Path(base_directory).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, storage_key: str) -> Path:
        validate_storage_key(storage_key)
        candidate = (self.base_dir / storage_key).resolve()
        if not candidate.is_relative_to(self.base_dir):
            raise BadRequestException(
                "Invalid storage key path",
                code="INVALID_STORAGE_KEY",
            )
        return candidate

    def save(self, storage_key: str, stream: BinaryReadableStream, content_type: str) -> None:
        _ = content_type
        target_path = self._resolve_safe_path(storage_key)
        if target_path.exists():
            raise ServiceUnavailableException(
                "Storage object key collision detected",
                code="STORAGE_COLLISION",
            )

        target_dir = target_path.parent
        temp_path = target_dir / f".tmp-{uuid.uuid4().hex}"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            stream.seek(0)
            with open(temp_path, "wb") as out_file:
                while True:
                    chunk = stream.read(STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
            os.replace(temp_path, target_path)
        except AppStoragePassThroughError:
            raise
        except Exception as exc:
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()
            logger.exception(
                "Failed to save file to local storage",
                extra={"extra_data": {"error_type": type(exc).__name__}},
            )
            raise ServiceUnavailableException(
                "Failed to store uploaded document",
                code="STORAGE_WRITE_FAILED",
            ) from exc

    def open_stream(self, storage_key: str) -> Iterator[bytes]:
        target_path = self._resolve_safe_path(storage_key)
        if not target_path.is_file():
            raise NotFoundException(
                "Stored document content was not found",
                code="STORAGE_OBJECT_NOT_FOUND",
            )

        def _file_iterator() -> Iterator[bytes]:
            with open(target_path, "rb") as file_obj:
                while True:
                    chunk = file_obj.read(STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk

        return _file_iterator()

    def delete(self, storage_key: str) -> None:
        target_path = self._resolve_safe_path(storage_key)
        try:
            if target_path.is_file():
                target_path.unlink()
        except OSError as exc:
            logger.exception(
                "Failed to delete file from local storage",
                extra={"extra_data": {"error_type": type(exc).__name__}},
            )
            raise ServiceUnavailableException(
                "Failed to delete stored document",
                code="STORAGE_DELETE_FAILED",
            ) from exc

    def exists(self, storage_key: str) -> bool:
        target_path = self._resolve_safe_path(storage_key)
        return target_path.is_file()


class AppStoragePassThroughError(Exception):
    """Marker class for internal storage exception handling."""


class S3StorageProvider:
    """S3-compatible object storage provider using boto3."""

    def __init__(
        self,
        *,
        bucket_name: str,
        region_name: str = "us-east-1",
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        s3_client: object | None = None,
    ) -> None:
        self.bucket_name = bucket_name
        if s3_client is not None:
            self.client = s3_client
        else:
            self.client = boto3.client(
                "s3",
                region_name=region_name,
                endpoint_url=endpoint_url or None,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            )

    def save(self, storage_key: str, stream: BinaryReadableStream, content_type: str) -> None:
        validate_storage_key(storage_key)
        try:
            stream.seek(0)
            self.client.upload_fileobj(  # type: ignore[attr-defined]
                stream,
                self.bucket_name,
                storage_key,
                ExtraArgs={"ContentType": content_type},
            )
        except (BotoCoreError, ClientError, OSError) as exc:
            logger.exception(
                "Failed to upload object to S3 storage",
                extra={"extra_data": {"error_type": type(exc).__name__}},
            )
            raise ServiceUnavailableException(
                "Failed to store uploaded document",
                code="STORAGE_WRITE_FAILED",
            ) from exc

    def open_stream(self, storage_key: str) -> Iterator[bytes]:
        validate_storage_key(storage_key)
        try:
            response = self.client.get_object(  # type: ignore[attr-defined]
                Bucket=self.bucket_name,
                Key=storage_key,
            )
            body = response["Body"]
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in ("404", "NoSuchKey", "NotFound"):
                raise NotFoundException(
                    "Stored document content was not found",
                    code="STORAGE_OBJECT_NOT_FOUND",
                ) from exc
            raise ServiceUnavailableException(
                "Failed to retrieve stored document",
                code="STORAGE_READ_FAILED",
            ) from exc
        except BotoCoreError as exc:
            raise ServiceUnavailableException(
                "Failed to retrieve stored document",
                code="STORAGE_READ_FAILED",
            ) from exc

        def _s3_iterator() -> Iterator[bytes]:
            try:
                while True:
                    chunk: bytes = body.read(STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk
            finally:
                body.close()

        return _s3_iterator()

    def delete(self, storage_key: str) -> None:
        validate_storage_key(storage_key)
        try:
            self.client.delete_object(  # type: ignore[attr-defined]
                Bucket=self.bucket_name,
                Key=storage_key,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.exception(
                "Failed to delete object from S3 storage",
                extra={"extra_data": {"error_type": type(exc).__name__}},
            )
            raise ServiceUnavailableException(
                "Failed to delete stored document",
                code="STORAGE_DELETE_FAILED",
            ) from exc

    def exists(self, storage_key: str) -> bool:
        validate_storage_key(storage_key)
        try:
            self.client.head_object(  # type: ignore[attr-defined]
                Bucket=self.bucket_name,
                Key=storage_key,
            )
            return True
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise ServiceUnavailableException(
                "Failed to check storage object status",
                code="STORAGE_CHECK_FAILED",
            ) from exc
        except BotoCoreError as exc:
            raise ServiceUnavailableException(
                "Failed to check storage object status",
                code="STORAGE_CHECK_FAILED",
            ) from exc


def create_storage_provider(settings: Settings | None = None) -> StorageProvider:
    cfg = settings or get_settings()
    if cfg.STORAGE_BACKEND == "s3":
        return S3StorageProvider(
            bucket_name=cfg.S3_BUCKET_NAME,
            region_name=cfg.S3_REGION,
            endpoint_url=cfg.S3_ENDPOINT_URL,
            access_key_id=cfg.S3_ACCESS_KEY_ID,
            secret_access_key=cfg.S3_SECRET_ACCESS_KEY,
        )
    return LocalStorageProvider(cfg.LOCAL_STORAGE_PATH)


def get_storage_provider() -> StorageProvider:
    return create_storage_provider()
