def command_help():
    print("Available commands:")
    for command, (args, func) in self._commands.items():
        print(f"- {command} {' '.join([f'<{arg}>' for arg in args]) if args else ''} --> {func.__doc__}")

def command_alias(mac, alias):
    # bind mac address to alias, store in file
    