"""Запуск DeepSeek Agent с автоматическим открытием браузера."""

import threading
import webbrowser
import socket

from app import app


def find_available_port(start: int = 5000, stop: int = 5100) -> int:
    """Возвращает первый свободный локальный порт в указанном диапазоне."""
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"Не найден свободный порт в диапазоне {start}–{stop}.")


def open_browser(url: str) -> None:
    webbrowser.open(url)


if __name__ == "__main__":
    port = find_available_port()
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.0, open_browser, args=(url,)).start()
    if port != 5000:
        print(f"Порт 5000 занят другой программой. Используется свободный порт {port}.")
    print(f"DeepSeek Agent запущен: {url}")
    print("Чтобы остановить сервер, нажмите Ctrl+C.\n")
    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    finally:
        app.extensions["whisper_service"].stop()
