import getpass

from app.auth.passwords import hash_password


def main() -> None:
    password = getpass.getpass("Administrator password: ")
    confirmation = getpass.getpass("Confirm administrator password: ")
    if not password:
        raise ValueError("password must not be empty")
    if password != confirmation:
        raise ValueError("password confirmation does not match")
    print(hash_password(password))


if __name__ == "__main__":
    main()
