import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError
from pwdlib import PasswordHash

_PASSWORD_HASH = PasswordHash.recommended()
# A fixed Argon2id hash ensures unknown usernames still perform equivalent work.
_DUMMY_PASSWORD_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$"
    "X1FuWEnyxJejzFgsc83NaA$"
    "Lo2AlDUvVRjnr/rMThrenvp0ffAxxQVj304q0rRY3RE"
)


def hash_password(password: str) -> str:
    return _PASSWORD_HASH.hash(password)


def is_valid_argon2id_hash(value: str) -> bool:
    if not value.startswith("$argon2id$"):
        return False
    try:
        PasswordHasher().check_needs_rehash(value)
    except InvalidHashError:
        return False
    return True


def verify_admin_credentials(
    *,
    submitted_username: str,
    submitted_password: str,
    configured_username: str,
    configured_password_hash: str,
) -> bool:
    username_matches = secrets.compare_digest(
        submitted_username.encode("utf-8"), configured_username.encode("utf-8")
    )
    selected_hash = (
        configured_password_hash if username_matches else _DUMMY_PASSWORD_HASH
    )
    try:
        password_matches = _PASSWORD_HASH.verify(submitted_password, selected_hash)
    except Exception:
        password_matches = False
    return username_matches and password_matches
