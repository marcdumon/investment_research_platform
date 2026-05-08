import json
import pathlib


def plotly_template() -> str:
    """Return 'plotly_dark' or 'plotly_white' based on VS Code settings.json."""
    candidates = [
        pathlib.Path.home() / '.config/Code/User/settings.json',
        pathlib.Path.home() / 'Library/Application Support/Code/User/settings.json',
        pathlib.Path.home() / 'AppData/Roaming/Code/User/settings.json',
    ]
    for p in candidates:
        if p.exists():
            try:
                theme = json.loads(p.read_text()).get('workbench.colorTheme', '')
                return 'plotly_white' if 'light' in theme.lower() else 'plotly_dark'
            except Exception:
                pass
    return 'plotly_dark'
