class NoKitException(Exception):
    def __init__(self, mac: str):
        self.mac = mac
        super().__init__(f"Kit {mac} not found in database.")


class BadgeNeededException(Exception):
    def __init__(self):
        super().__init__(f"You need to authenticate with your badge first.")


class UnresponsiveKitException(Exception):
    def __init__(self):
        super().__init__(f"Kit is unresponsive or sleeping, try badging it.")


class RegistrationTimeoutException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class UserNotFoundException(Exception):
    def __init__(self, user: str):
        self.user = user
        super().__init__(f"User {user} not found in database.")


class CompanyNotFoundException(Exception):
    def __init__(self, company_id: int):
        self.company_id = company_id
        super().__init__(f"Company {company_id} not found in database.")
