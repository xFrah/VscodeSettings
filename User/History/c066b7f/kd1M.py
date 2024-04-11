import paho.mqtt.client as mqtt
import time
import threading
import json
import os
import backend_commands as bc


class CMD:
    def __init__(self):
        self._commands_list = {
            "help": ((), "Display this help message."),
            "alias": (
                (
                    "add:remove",
                    "MAC",
                    "alias",
                ),
                "Add or remove an alias for a MAC address.",
            ),
            "clearalias": (("MAC",), "Clear all aliases for a MAC address."),
            "list": ((), "List all known MAC addresses, with their aliases."),
            "kit": (("add:remove", "mac|alias"), "Add or remove a kit."),
            "kitlist": ((), "List all known kits."),
        }
        self._commands_queue = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self.input_thread, daemon=True)
        self._thread.start()

    # read thread
    def input_thread(self):
        while True:
            string = input(">>>")
            try:
                command, *args = string.split()
            except Exception as e:
                print(f"Couldn't parse command: {e}")
                continue

            if command == "help":
                self.command_help()
            elif command not in self._commands_list:
                print(f"Unknown command: {command}")
            elif len(args) < len(self._commands_list[command][0]):
                n_args = len(self._commands_list[command][0])
                missing_arguments = self._commands_list[command][0][len(args) :]
                print(f"Missing arguments for command {command}: {' '.join([f'<{arg}>' for arg in missing_arguments])}")
            else:
                self.push_command(command, args)
            time.sleep(0.1)

    def command_ready(self):
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


def main():
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)
    client.loop_start()
    # debug messages
    client.on_log = lambda client, userdata, level, buf: print(f"LOG: {buf}", end="\n>>>")
    cmd = CMD()

    while True:
        if cmd.command_ready():
            command, args = cmd.get_command()
            if command == "alias":
                bc.command_alias(args)
            elif command == "clearalias":
                bc.command_clearalias(args)
            elif command == "list":
                bc.command_list()
            elif command == "kit":
                bc.command_kit(args)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
