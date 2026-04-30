from importlib import import_module
from pathlib import Path

for _p in sorted(Path(__file__).parent.glob("*.py")):
    if _p.stem != "__init__":
        import_module(f".{_p.stem}", package=__name__)
