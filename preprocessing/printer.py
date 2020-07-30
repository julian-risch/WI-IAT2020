from colorama import init, Fore

init()


def _print_colorful(color, text):
    print(f'{color}{text}{Fore.RESET}')


def print_error(text):
    _print_colorful(Fore.RED, text)


def print_success(text):
    _print_colorful(Fore.GREEN, text)


def print_progress(text):
    _print_colorful(Fore.LIGHTWHITE_EX, text)


def print_warning( text):
    _print_colorful(Fore.YELLOW, text)