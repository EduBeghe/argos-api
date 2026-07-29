import pytest

from app.auth.passwords import hash_password, verify_admin_credentials
from app.jobs import hash_admin_password


def test_correct_and_incorrect_password_verification() -> None:
    password_hash = hash_password("correct-password")
    assert verify_admin_credentials(
        submitted_username="admin",
        submitted_password="correct-password",
        configured_username="admin",
        configured_password_hash=password_hash,
    )
    assert not verify_admin_credentials(
        submitted_username="admin",
        submitted_password="wrong-password",
        configured_username="admin",
        configured_password_hash=password_hash,
    )


def test_unknown_username_uses_dummy_hash_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def verify(password: str, password_hash: str) -> bool:
        del password
        calls.append(password_hash)
        return False

    configured_hash = hash_password("password")
    monkeypatch.setattr("app.auth.passwords._PASSWORD_HASH.verify", verify)
    assert not verify_admin_credentials(
        submitted_username="unknown",
        submitted_password="password",
        configured_username="admin",
        configured_password_hash=configured_hash,
    )
    assert len(calls) == 1
    assert calls[0] != configured_hash


@pytest.mark.parametrize("passwords", [("", ""), ("first", "second")])
def test_hash_helper_rejects_empty_or_mismatched_confirmation(
    monkeypatch: pytest.MonkeyPatch, passwords: tuple[str, str]
) -> None:
    entries = iter(passwords)
    monkeypatch.setattr(
        "app.jobs.hash_admin_password.getpass.getpass", lambda _prompt: next(entries)
    )
    with pytest.raises(ValueError):
        hash_admin_password.main()


def test_hash_helper_prints_only_argon2id_hash(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = iter(("correct-password", "correct-password"))
    monkeypatch.setattr(
        "app.jobs.hash_admin_password.getpass.getpass", lambda _prompt: next(entries)
    )
    hash_admin_password.main()
    output = capsys.readouterr().out.strip()
    assert output.startswith("$argon2id$")
    assert "correct-password" not in output
