import argparse
import sys

import pytest


def cmd_test(args: argparse.Namespace) -> int:
    pytest_args = args.pytest_args or ["tests/"]
    if args.verbose:
        pytest_args = ["-v", *pytest_args]
    if args.integration:
        pytest_args = ["--integration", *pytest_args]
    return pytest.main(pytest_args)


def main() -> None:
    parser = argparse.ArgumentParser(prog="irp")
    sub = parser.add_subparsers(dest="command", required=True)

    p_test = sub.add_parser("test", help="run test suite")
    p_test.add_argument("-v", "--verbose", action="store_true")
    p_test.add_argument("--integration", action="store_true", help="include integration tests")
    p_test.add_argument("pytest_args", nargs="*", help="extra args passed to pytest")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
