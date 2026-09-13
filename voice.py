"""Постоянный локальный whisper.cpp-сервис для голосового ввода."""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".mp4"}


class WhisperService:
    """Запускает whisper-server один раз и держит модель загруженной до выхода."""

    def __init__(self, project_dir: Path, autostart: bool = True) -> None:
        self.project_dir = project_dir
        self.server_path = Path(os.getenv(
            "WHISPER_SERVER_PATH",
            project_dir / "tools" / "whisper" / "bin" / "whisper-server.exe",
        ))
        self.model_path = Path(os.getenv(
            "WHISPER_MODEL_PATH",
            project_dir / "models" / "ggml-large-v3-turbo.bin",
        ))
        self.ffmpeg_dir = Path(os.getenv(
            "WHISPER_FFMPEG_DIR",
            project_dir / "tools" / "ffmpeg" / "bin",
        ))
        self.host = "127.0.0.1"
        configured_port = os.getenv("WHISPER_PORT")
        self.port = int(configured_port) if configured_port else self._available_port(8091, 8100)
        self.language = os.getenv("WHISPER_LANGUAGE", "ru")
        self._process: subprocess.Popen[str] | None = None
        self._start_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._phase = "stopped"
        self._message = "Whisper ещё не запущен."
        self._last_log = ""
        atexit.register(self.stop)
        if autostart and os.getenv("WHISPER_AUTOSTART", "true").lower() not in {"0", "false", "no"}:
            self.start()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def _available_port(first: int, last: int) -> int:
        for port in range(first, last + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise RuntimeError(f"Нет свободного порта для Whisper в диапазоне {first}-{last}.")

    def start(self) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                return
            if self._start_thread and self._start_thread.is_alive():
                return
            missing = []
            if not self.server_path.is_file():
                missing.append(f"не найден {self.server_path.name}")
            if not self.model_path.is_file():
                missing.append(f"не найдена модель {self.model_path.name}")
            if missing:
                self._phase = "not_installed"
                self._message = "Whisper не установлен: " + ", ".join(missing) + "."
                return
            self._phase = "starting"
            self._message = "Загружаем large-v3-turbo в память RX 6600…"
            self._start_thread = threading.Thread(target=self._start_worker, daemon=True)
            self._start_thread.start()

    def _start_worker(self) -> None:
        environment = os.environ.copy()
        if self.ffmpeg_dir.is_dir():
            environment["PATH"] = str(self.ffmpeg_dir) + os.pathsep + environment.get("PATH", "")
        # whisper.cpp on Windows still passes the model name through a narrow
        # native string. A relative ASCII-only path keeps projects in folders
        # such as "Задание 9" working correctly.
        model_argument = os.path.relpath(self.model_path, self.server_path.parent)
        command = [
            str(self.server_path),
            "--host", self.host,
            "--port", str(self.port),
            "--model", model_argument,
            "--language", self.language,
            "--convert",
        ]
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                command,
                cwd=self.server_path.parent,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )
            with self._lock:
                self._process = process
            threading.Thread(target=self._capture_output, args=(process,), daemon=True).start()
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(self._last_log or f"whisper-server завершился с кодом {process.returncode}")
                try:
                    with socket.create_connection((self.host, self.port), timeout=1.5):
                        with self._lock:
                            self._phase = "ready"
                            self._message = "Whisper готов · Vulkan · RX 6600 · large-v3-turbo"
                        return
                except OSError:
                    pass
                time.sleep(0.5)
            raise RuntimeError("модель не загрузилась за 180 секунд")
        except Exception as error:
            with self._lock:
                self._phase = "error"
                self._message = f"Не удалось запустить Whisper: {error}"

    def _capture_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            stripped = line.strip()
            if stripped:
                with self._lock:
                    self._last_log = stripped[-500:]
        if self._phase == "ready" and process.poll() is not None:
            with self._lock:
                self._phase = "error"
                self._message = f"Whisper неожиданно завершился с кодом {process.returncode}."

    def status(self) -> dict[str, Any]:
        with self._lock:
            process_running = bool(self._process and self._process.poll() is None)
            return {
                "phase": self._phase,
                "ready": self._phase == "ready" and process_running,
                "message": self._message,
                "model": "large-v3-turbo",
                "backend": "Vulkan",
                "language": self.language,
                "server_path": str(self.server_path),
                "model_path": str(self.model_path),
                "last_log": self._last_log if self._phase == "error" else "",
            }

    def transcribe(self, audio: bytes, filename: str, content_type: str) -> dict[str, Any]:
        if not audio:
            raise ValueError("Получена пустая аудиозапись.")
        if len(audio) > MAX_AUDIO_BYTES:
            raise ValueError("Аудиозапись больше 25 МБ. Запишите более короткое сообщение.")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError("Неподдерживаемый формат аудио.")
        if not self.status()["ready"]:
            raise RuntimeError(self.status()["message"])

        boundary = uuid.uuid4().hex
        safe_name = Path(filename).name.replace('"', "")
        fields = {
            "response_format": "json",
            "language": self.language,
            "temperature": "0.0",
            "temperature_inc": "0.2",
        }
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
            )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
            f'Content-Type: {content_type or "application/octet-stream"}\r\n\r\n'.encode()
        )
        parts.extend([audio, b"\r\n", f"--{boundary}--\r\n".encode()])
        request_body = b"".join(parts)
        inference_request = urllib.request.Request(
            self.base_url + "/inference",
            data=request_body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(inference_request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Whisper не смог распознать запись: {error}") from error

        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("В записи не удалось распознать речь.")
        return {"text": text, "language": self.language}

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            self._phase = "stopped"
            self._message = "Whisper остановлен, видеопамять освобождена."
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
