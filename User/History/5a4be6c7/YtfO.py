class NoKitException(Exception):
    def __init__(self, mac: str):
        self.mac = mac
        super().__init__(f"Kit {mac} not found in database.")


class InvalidMACException(Exception):
    def __init__(self, mac: str, message: str = ""):
        self.mac = mac
        super().__init__(f"Invalid MAC address {mac}." if message == "" else message)


class BadgeNeededException(Exception):
    def __init__(self):
        super().__init__(f"You need to authenticate with your badge first.")


class UnresponsiveKitException(Exception):
    def __init__(self):
        super().__init__(f"Kit is unresponsive or sleeping, try badging it.")


class RegistrationTimeoutException(Exception):
    def __init__(self):
        super().__init__(f"Timeout while waiting for unregistered items.")


class UserNotFoundException(Exception):
    def __init__(self, user: str):
        self.user = user
        super().__init__(f"User {user} not found in database.")


class InvalidExpirationDateException(Exception):
    def __init__(self, date: str):
        self.date = date
        super().__init__(f"Invalid expiration date {date}.")


class UnknownItemTypeException(Exception):
    def __init__(self, item_type: str):
        self.item_type = item_type
        super().__init__(f"Unknown item type {item_type}.")


class CompanyNotFoundException(Exception):
    def __init__(self, company_id: int):
        self.company_id = company_id
        super().__init__(f"Company {company_id} not found in database.")
