import os
import json
import hashlib


# ============================================================
# AUTHENTICATION CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

USERS_FILE = os.path.join(
    BASE_DIR,
    "users.json"
)


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password):
    """
    Convert a password into a SHA-256 hash.
    """

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# LOAD USERS
# ============================================================

def load_users():

    if not os.path.exists(USERS_FILE):

        return {}

    try:

        with open(
            USERS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):

        return {}


# ============================================================
# SAVE USERS
# ============================================================

def save_users(users):

    with open(
        USERS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            users,
            file,
            indent=4
        )


# ============================================================
# CREATE USER
# ============================================================

def create_user(
    email,
    password
):

    email = email.strip().lower()

    users = load_users()

    if email in users:

        return False, "User already exists."

    if not email:

        return False, "Email is required."

    if len(password) < 6:

        return False, (
            "Password must contain at least 6 characters."
        )

    users[email] = {
        "password": hash_password(password)
    }

    save_users(users)

    return True, "Account created successfully."


# ============================================================
# LOGIN
# ============================================================

def login_user(
    email,
    password
):

    email = email.strip().lower()

    users = load_users()

    if email not in users:

        return False

    stored_password = users[email][
        "password"
    ]

    entered_password = hash_password(
        password
    )

    if stored_password == entered_password:

        return True

    return False


# ============================================================
# CHECK USER
# ============================================================

def user_exists(email):

    email = email.strip().lower()

    users = load_users()

    return email in users


# ============================================================
# DELETE USER
# ============================================================

def delete_user(email):

    email = email.strip().lower()

    users = load_users()

    if email not in users:

        return False

    del users[email]

    save_users(users)

    return True


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_email = "test@example.com"
    test_password = "test123"

    created, message = create_user(
        test_email,
        test_password
    )

    print(message)

    if login_user(
        test_email,
        test_password
    ):

        print("Login test: SUCCESS")

    else:

        print("Login test: FAILED")