"""Secure user authentication module for the AI-Powered Healthcare
Diagnosis Assistant.

This module provides reusable, framework-independent functions to
register users, authenticate login attempts, and handle logout state.
User records are persisted to a JSON file on disk at
``data/users.json``, with passwords hashed using SHA-256. The data
directory and file are created automatically if they do not exist.

Typical usage example:
    from src.auth import register_user, login_user, logout_user

    register_user("john_doe", "SecurePass123")
    is_authenticated = login_user("john_doe", "SecurePass123")
    logout_user()
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MIN_USERNAME_LENGTH: Final[int] = 3
MAX_USERNAME_LENGTH: Final[int] = 30
MIN_PASSWORD_LENGTH: Final[int] = 8
USERNAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_.]+$")

DATA_DIR: Final[Path] = Path("data")
USERS_FILE_PATH: Final[Path] = DATA_DIR / "users.json"


class AuthenticationError(Exception):
    """Raised when a user authentication operation fails.

    This exception is raised whenever registration or login input fails
    validation, a username is already taken, login credentials are
    invalid, or the user data store cannot be read or written.
    """


def _hash_password(password: str) -> str:
    """Hash a plaintext password using SHA-256.

    Args:
        password: The plaintext password to hash.

    Returns:
        The hexadecimal SHA-256 digest of the password.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _validate_username(username: str) -> str:
    """Validate and normalize a username value.

    Args:
        username: The username to validate.

    Returns:
        The stripped, validated username.

    Raises:
        AuthenticationError: If the username is empty, out of the
            allowed length range, or contains disallowed characters.
    """
    if not isinstance(username, str) or not username.strip():
        message = "Username cannot be empty."
        logger.error(message)
        raise AuthenticationError(message)

    normalized_username = username.strip()

    if not (MIN_USERNAME_LENGTH <= len(normalized_username) <= MAX_USERNAME_LENGTH):
        message = (
            f"Username must be between {MIN_USERNAME_LENGTH} and "
            f"{MAX_USERNAME_LENGTH} characters long."
        )
        logger.error(message)
        raise AuthenticationError(message)

    if not USERNAME_PATTERN.match(normalized_username):
        message = (
            "Username may only contain letters, numbers, underscores, "
            "and periods."
        )
        logger.error(message)
        raise AuthenticationError(message)

    return normalized_username


def _validate_password(password: str) -> str:
    """Validate a password value.

    Args:
        password: The password to validate.

    Returns:
        The validated password.

    Raises:
        AuthenticationError: If the password is empty or shorter than
            the minimum required length.
    """
    if not isinstance(password, str) or not password:
        message = "Password cannot be empty."
        logger.error(message)
        raise AuthenticationError(message)

    if len(password) < MIN_PASSWORD_LENGTH:
        message = f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        logger.error(message)
        raise AuthenticationError(message)

    return password


def _ensure_users_file_exists(
    data_dir: Path = DATA_DIR,
    users_file_path: Path = USERS_FILE_PATH,
) -> None:
    """Ensure the data directory and users file exist on disk.

    Args:
        data_dir: The directory that should contain the users file.
        users_file_path: The path to the users JSON file.

    Raises:
        AuthenticationError: If the directory or file cannot be created.
    """
    try:
        data_dir.mkdir(parents=True, exist_ok=True)

        if not users_file_path.exists():
            users_file_path.write_text(json.dumps({}), encoding="utf-8")
            logger.info("Created new users file at: %s", users_file_path)
    except OSError as exc:
        message = f"Failed to initialize user data storage at: {users_file_path}"
        logger.error(message)
        raise AuthenticationError(message) from exc


def _load_users(users_file_path: Path = USERS_FILE_PATH) -> dict[str, str]:
    """Load the users dictionary from the JSON data store.

    If the JSON file is corrupted or contains invalid data, it is safely
    reset to an empty users dictionary rather than raising an error.

    Args:
        users_file_path: The path to the users JSON file.

    Returns:
        A dictionary mapping usernames to hashed passwords.

    Raises:
        AuthenticationError: If the users file cannot be read due to an
            I/O error.
    """
    _ensure_users_file_exists(users_file_path.parent, users_file_path)

    logger.info("Loading users from: %s", users_file_path)

    try:
        raw_content = users_file_path.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"Failed to read user data from: {users_file_path}"
        logger.error(message)
        raise AuthenticationError(message) from exc

    try:
        users = json.loads(raw_content) if raw_content.strip() else {}
    except json.JSONDecodeError:
        logger.warning(
            "Users file at %s is corrupted. Resetting to empty user store.",
            users_file_path,
        )
        users = {}
        _save_users(users, users_file_path)

    if not isinstance(users, dict):
        logger.warning(
            "Users file at %s contains invalid data. Resetting to empty "
            "user store.",
            users_file_path,
        )
        users = {}
        _save_users(users, users_file_path)

    return users


def _save_users(
    users: dict[str, str],
    users_file_path: Path = USERS_FILE_PATH,
) -> None:
    """Persist the users dictionary to the JSON data store.

    Args:
        users: The users dictionary to save.
        users_file_path: The path to the users JSON file.

    Raises:
        AuthenticationError: If the users file cannot be written.
    """
    logger.info("Saving users to: %s", users_file_path)

    try:
        users_file_path.parent.mkdir(parents=True, exist_ok=True)
        users_file_path.write_text(
            json.dumps(users, indent=4), encoding="utf-8"
        )
    except OSError as exc:
        message = f"Failed to save user data to: {users_file_path}"
        logger.error(message)
        raise AuthenticationError(message) from exc

    logger.info("User data saved successfully to: %s", users_file_path)


def register_user(username: str, password: str) -> bool:
    """Register a new user with a hashed password.

    The registered user is persisted to ``data/users.json``.

    Args:
        username: The desired username. Must be unique, between 3 and 30
            characters, and contain only letters, numbers, underscores,
            or periods.
        password: The plaintext password. Must be at least 8 characters
            long.

    Returns:
        True if registration was successful.

    Raises:
        AuthenticationError: If the username or password is invalid, if
            the username is already registered, or if the user data
            cannot be persisted.
    """
    validated_username = _validate_username(username)
    validated_password = _validate_password(password)

    users = _load_users()

    if validated_username in users:
        message = f"Username '{validated_username}' is already taken."
        logger.error("Registration failed: %s", message)
        raise AuthenticationError(message)

    users[validated_username] = _hash_password(validated_password)
    _save_users(users)

    logger.info("Registration successful for user '%s'.", validated_username)
    return True


def login_user(username: str, password: str) -> bool:
    """Authenticate a user login attempt against persisted user data.

    Args:
        username: The username to authenticate.
        password: The plaintext password to verify.

    Returns:
        True if the username exists and the password matches.

    Raises:
        AuthenticationError: If the username or password is invalid, if
            the username is not registered, or if the password does not
            match.
    """
    validated_username = _validate_username(username)
    validated_password = _validate_password(password)

    users = _load_users()
    stored_password_hash = users.get(validated_username)

    if stored_password_hash is None:
        message = f"No account found for username '{validated_username}'."
        logger.error("Login failed: %s", message)
        raise AuthenticationError(message)

    if stored_password_hash != _hash_password(validated_password):
        message = "Invalid username or password."
        logger.error("Login failed for user '%s': %s", validated_username, message)
        raise AuthenticationError(message)

    logger.info("Login successful for user '%s'.", validated_username)
    return True


def logout_user() -> bool:
    """Log out the currently authenticated user.

    Returns:
        True, indicating the logout operation completed successfully.
    """
    logger.info("User logged out successfully.")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        register_user(username="john_doe", password="SecurePass123")
        logger.info("Registration demonstration successful.")

        is_logged_in = login_user(username="john_doe", password="SecurePass123")
        logger.info("Login demonstration successful: %s", is_logged_in)

        is_logged_out = logout_user()
        logger.info("Logout demonstration successful: %s", is_logged_out)
    except AuthenticationError:
        logger.exception("Authentication demonstration failed.")
