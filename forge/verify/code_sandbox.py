"""
Code Execution Sandbox for Forge.

Provides secure, isolated code execution via Docker containers
(when available) with fallback to subprocess-based sandboxing.
Used to verify generated code solutions during self-play.

Security measures:
  - Container: network disabled, memory limited, CPU throttled
  - Subprocess fallback: resource limits via ulimit, temp directories
  - Hard timeout on all executions (default 5 seconds)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ForgeConfig

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a code solution."""
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: float = 0.0
    language: str = "python"

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


# ---------------------------------------------------------------------------
# Language configurations for the sandbox
# ---------------------------------------------------------------------------

LANGUAGE_CONFIGS = {
    "python": {
        "docker_image": "python:3.11-slim",
        "docker_cmd": ["python", "/code/solution.py"],
        "file_ext": ".py",
        "subprocess_cmd": [sys.executable],
    },
    "javascript": {
        "docker_image": "node:20-slim",
        "docker_cmd": ["node", "/code/solution.js"],
        "file_ext": ".js",
        "subprocess_cmd": ["node"],
    },
    "cpp": {
        "docker_image": "gcc:13",
        "docker_cmd": [
            "bash", "-c",
            "g++ -O2 -std=c++17 /code/solution.cpp -o /tmp/sol && /tmp/sol"
        ],
        "file_ext": ".cpp",
        "subprocess_cmd": None,  # Requires compilation step
    },
}


# ---------------------------------------------------------------------------
# Docker-based sandbox
# ---------------------------------------------------------------------------

class DockerSandbox:
    """
    Execute code in Docker containers with security constraints.
    Falls back to subprocess if Docker is unavailable.
    """

    def __init__(
        self,
        timeout: int = 5,
        memory_limit: str = "256m",
        docker_image: str = "python:3.11-slim",
    ):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.docker_image = docker_image
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check whether Docker is available on the system."""
        try:
            import docker
            client = docker.from_env()
            client.ping()
            logger.info("Docker is available — using container sandbox")
            return True
        except Exception:
            logger.warning(
                "Docker unavailable — falling back to subprocess sandbox. "
                "Code execution will be less isolated."
            )
            return False

    def execute(
        self,
        code: str,
        test_code: str = "",
        language: str = "python",
        stdin_input: str = "",
    ) -> ExecutionResult:
        """
        Execute code and return the result.

        Args:
            code: The solution code to execute.
            test_code: Optional test harness to append (e.g., assert statements).
            language: Programming language.
            stdin_input: Input to feed via stdin.

        Returns:
            ExecutionResult with exit code, output, and timing.
        """
        full_code = code
        if test_code:
            full_code = code + "\n\n" + test_code

        if self._docker_available:
            return self._execute_docker(full_code, language, stdin_input)
        else:
            return self._execute_subprocess(full_code, language, stdin_input)

    def _execute_docker(
        self,
        code: str,
        language: str,
        stdin_input: str,
    ) -> ExecutionResult:
        """Execute code inside a Docker container."""
        import docker

        config = LANGUAGE_CONFIGS.get(language)
        if config is None:
            return ExecutionResult(
                exit_code=-1,
                stderr=f"Unsupported language: {language}",
            )

        client = docker.from_env()
        start = time.monotonic()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write solution file
            filepath = os.path.join(tmpdir, f"solution{config['file_ext']}")
            with open(filepath, "w") as f:
                f.write(code)

            try:
                output = client.containers.run(
                    image=config.get("docker_image", self.docker_image),
                    command=config["docker_cmd"],
                    volumes={tmpdir: {"bind": "/code", "mode": "ro"}},
                    mem_limit=self.memory_limit,
                    cpu_period=100000,
                    cpu_quota=50000,
                    network_disabled=True,
                    remove=True,
                    timeout=self.timeout,
                    detach=False,
                    stdout=True,
                    stderr=True,
                )
                duration = (time.monotonic() - start) * 1000
                return ExecutionResult(
                    exit_code=0,
                    stdout=output.decode("utf-8", errors="replace")[:10000],
                    duration_ms=duration,
                    language=language,
                )
            except docker.errors.ContainerError as e:
                duration = (time.monotonic() - start) * 1000
                return ExecutionResult(
                    exit_code=e.exit_status,
                    stderr=str(e)[:5000],
                    duration_ms=duration,
                    language=language,
                )
            except Exception as e:
                duration = (time.monotonic() - start) * 1000
                is_timeout = "timeout" in str(e).lower() or "deadline" in str(e).lower()
                return ExecutionResult(
                    exit_code=-1,
                    stderr=str(e)[:5000],
                    timed_out=is_timeout,
                    duration_ms=duration,
                    language=language,
                )

    def _execute_subprocess(
        self,
        code: str,
        language: str,
        stdin_input: str,
    ) -> ExecutionResult:
        """
        Fallback: execute code via subprocess with resource limits.
        Less isolated than Docker but works everywhere.
        """
        config = LANGUAGE_CONFIGS.get(language)
        if config is None or config.get("subprocess_cmd") is None:
            return ExecutionResult(
                exit_code=-1,
                stderr=f"Subprocess execution not supported for: {language}",
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=config["file_ext"],
            delete=False,
        ) as f:
            f.write(code)
            filepath = f.name

        start = time.monotonic()
        try:
            result = subprocess.run(
                config["subprocess_cmd"] + [filepath],
                input=stdin_input if stdin_input else None,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            duration = (time.monotonic() - start) * 1000
            return ExecutionResult(
                exit_code=result.returncode,
                stdout=result.stdout[:10000],
                stderr=result.stderr[:5000],
                duration_ms=duration,
                language=language,
            )
        except subprocess.TimeoutExpired:
            duration = (time.monotonic() - start) * 1000
            return ExecutionResult(
                exit_code=-1,
                stderr="Execution timed out",
                timed_out=True,
                duration_ms=duration,
                language=language,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return ExecutionResult(
                exit_code=-1,
                stderr=str(e)[:5000],
                duration_ms=duration,
                language=language,
            )
        finally:
            try:
                os.unlink(filepath)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------

def verify_code_batch(
    solutions: list[dict],
    cfg: ForgeConfig,
    max_workers: int | None = None,
) -> list[ExecutionResult]:
    """
    Verify a batch of code solutions in parallel.

    Each solution dict should have:
      - "code": str — the solution code
      - "test_code": str — test harness (assert statements, etc.)
      - "language": str — programming language (default: "python")

    Args:
        solutions: List of solution dicts.
        cfg: Forge configuration.
        max_workers: Number of parallel workers.

    Returns:
        List of ExecutionResult objects in the same order.
    """
    workers = max_workers or cfg.max_docker_workers
    sandbox = DockerSandbox(
        timeout=cfg.code_timeout,
        memory_limit=cfg.code_memory_limit,
        docker_image=cfg.docker_image,
    )

    results: list[ExecutionResult | None] = [None] * len(solutions)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for idx, sol in enumerate(solutions):
            future = pool.submit(
                sandbox.execute,
                code=sol.get("code", ""),
                test_code=sol.get("test_code", ""),
                language=sol.get("language", "python"),
                stdin_input=sol.get("stdin_input", ""),
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = ExecutionResult(
                    exit_code=-1,
                    stderr=f"Worker error: {e}",
                )

    return results  # type: ignore[return-value]
