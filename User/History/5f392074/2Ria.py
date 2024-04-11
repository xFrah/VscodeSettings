import threading
import time


class CMD:
    def __init__(self):
        self._commands_list = {
            "help": ((), "Display this help message."),
            "ping": ((), "It pongs."),
            "testreport": (("company_id",), "Tests report function."),
            "company": (("add:remove", "company_name|company_id"), "Add or remove a company."),
            "listcompanies": ((), "List all known companies."),
            "kit": (("add:remove", "mac", "company_id"), "Add or remove a kit."),
            "listkits": ((), "List all known kits."),
            "adduser": (
                ("username", "password", "badge_id|NULL", "clearance", "company_id"),
                "Add a user to the database.",
            ),
            "removeuser": (("username", "company_id"), "Remove a user from the database."),
            "listuser": ((), "List all known users."),
        }
        self._commands_queue = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self.input_thread, daemon=True)
        self._thread.start()

    def input_thread(self):
        while True:
            try:
                string = input()
                command, *args = string.split()
                command = command.lower()

                if command == "help":
                    self.command_help()
                elif command not in self._commands_list:
                    print(f"Unknown command: {command}")
                elif len(args) < len(self._commands_list[command][0]):
                    missing_arguments = self._commands_list[command][0][len(args) :]
                    print(f"Missing arguments for command {command}: {' '.join([f'<{arg}>' for arg in missing_arguments])}")
                else:
                    self.push_command(command, args)
            except Exception as e:
                print(f"{e}")
            time.sleep(0.1)

    def data_ready(self):
        return len(self._commands_queue) > 0

    def get_command(self):
        with self._lock:
            return self._commands_queue.pop(0)

    def push_command(self, command, args):
        with self._lock:
            self._commands_queue.append((command, args))

    def command_help(self):
        print("Available commands:")
        for command, (args, description) in self._commands_list.items():
            print(f"- {command} {' '.join([f'<{arg}>' for arg in args]) if args else ''} --> {description}")
        print()
