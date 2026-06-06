def run_server() -> None:
    from irp.core.logging import configure_logging
    from irp.ui.app import app
    configure_logging()
    app.run(debug=True, dev_tools_hot_reload=False)
