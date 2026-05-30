def run_server() -> None:
    from irp.ui.app import app
    from irp.core.logging import _configure_logging as configure_logging
    configure_logging()
    app.run(debug=True, dev_tools_hot_reload=False)
