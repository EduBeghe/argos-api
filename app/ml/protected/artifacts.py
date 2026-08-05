import hashlib
import pickle
import zlib
from dataclasses import dataclass

from .contracts import ARTIFACT_FORMAT_VERSION, MAXIMUM_ARTIFACT_BYTES

TRUSTED_PROVENANCE = "argos_database_final_validation_v1"


class ArtifactIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SerializedArtifact:
    payload: bytes
    sha256: str
    byte_length: int
    format_version: int = ARTIFACT_FORMAT_VERSION
    compression: str = "zlib"


def serialize_trusted_artifact(
    value: object, *, maximum_bytes: int = MAXIMUM_ARTIFACT_BYTES
) -> SerializedArtifact:
    compressed = zlib.compress(pickle.dumps(value, protocol=5), level=9)
    if len(compressed) > maximum_bytes:
        raise ArtifactIntegrityError("model artifact exceeds configured maximum size")
    return SerializedArtifact(
        compressed, hashlib.sha256(compressed).hexdigest(), len(compressed)
    )


def load_trusted_artifact(
    payload: bytes,
    *,
    expected_sha256: str,
    provenance: str,
    format_version: int,
    stored_library_versions: dict[str, str],
    current_library_versions: dict[str, str],
) -> object:
    if provenance != TRUSTED_PROVENANCE:
        raise ArtifactIntegrityError("model artifact provenance is not trusted")
    if format_version != ARTIFACT_FORMAT_VERSION:
        raise ArtifactIntegrityError("model artifact format is incompatible")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ArtifactIntegrityError("model artifact SHA-256 mismatch")
    if stored_library_versions != current_library_versions:
        raise ArtifactIntegrityError("model artifact library versions are incompatible")
    try:
        return pickle.loads(zlib.decompress(payload))
    except (pickle.PickleError, zlib.error, EOFError) as error:
        raise ArtifactIntegrityError("model artifact payload is corrupt") from error
