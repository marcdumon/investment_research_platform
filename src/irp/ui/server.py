def run_server() -> None:
    from irp.ui.app import app
    from irp.core.logging import configure_logging
    configure_logging()
    app.run(debug=False)
