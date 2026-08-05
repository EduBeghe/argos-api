import pytest

from app.ml.protected.artifacts import (
    TRUSTED_PROVENANCE,
    ArtifactIntegrityError,
    load_trusted_artifact,
    serialize_trusted_artifact,
)


def test_trusted_artifact_round_trip_and_integrity() -> None:
    serialized = serialize_trusted_artifact({"model": [1, 2, 3]})
    restored = load_trusted_artifact(
        serialized.payload,
        expected_sha256=serialized.sha256,
        provenance=TRUSTED_PROVENANCE,
        format_version=serialized.format_version,
        stored_library_versions={"engine": "1"},
        current_library_versions={"engine": "1"},
    )
    assert restored == {"model": [1, 2, 3]}
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_trusted_artifact(
            serialized.payload + b"x",
            expected_sha256=serialized.sha256,
            provenance=TRUSTED_PROVENANCE,
            format_version=serialized.format_version,
            stored_library_versions={},
            current_library_versions={},
        )


def test_artifact_size_and_provenance_are_fail_closed() -> None:
    with pytest.raises(ArtifactIntegrityError, match="maximum"):
        serialize_trusted_artifact(bytes(range(256)) * 20, maximum_bytes=10)
    value = serialize_trusted_artifact("safe")
    with pytest.raises(ArtifactIntegrityError, match="provenance"):
        load_trusted_artifact(
            value.payload,
            expected_sha256=value.sha256,
            provenance="external",
            format_version=value.format_version,
            stored_library_versions={},
            current_library_versions={},
        )
