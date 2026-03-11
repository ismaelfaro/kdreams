"""Local compute backend — Phase 1 primary backend."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from kdream.backends.base import AbstractBackend
from kdream.core.recipe import ModelDescriptor, Recipe
from kdream.exceptions import BackendError, ModelDownloadError

console = Console()


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

class HardwareDetector:
    """Detect available compute hardware."""

    def detect(self) -> dict[str, Any]:
        """Return hardware info dict with keys: device, vram_gb, cuda_version."""
        info: dict[str, Any] = {"device": "cpu", "vram_gb": 0, "cuda_version": None}

        # Try CUDA
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                info["device"] = "cuda"
                info["vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                )
                info["cuda_version"] = torch.version.cuda
                return info
        except ImportError:
            pass

        # Try MPS (Apple Silicon)
        try:
            import torch  # type: ignore[import]
            if platform.processor() == "arm" and sys.platform == "darwin":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    info["device"] = "mps"
                    return info
        except ImportError:
            pass

        # Try nvidia-smi as fallback
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                vram_mib = int(result.stdout.strip().split("\n")[0])
                info["device"] = "cuda"
                info["vram_gb"] = round(vram_mib / 1024, 1)
        except Exception:
            pass

        return info


# ---------------------------------------------------------------------------
# Environment manager
# ---------------------------------------------------------------------------

class EnvironmentManager:
    """Manages repository cloning and UV virtual environments."""

    def clone_repo(self, repo_url: str, ref: str, dest: Path) -> None:
        """Clone *repo_url* at *ref* to *dest* (idempotent)."""
        if dest.exists() and (dest / ".git").exists():
            console.print(f"  [dim]Repo already cloned at {dest}[/dim]")
            return

        console.print(f"  Cloning [cyan]{repo_url}[/cyan] @ {ref} ...")
        try:
            import git  # type: ignore[import]
            git.Repo.clone_from(repo_url, dest, depth=1, branch=ref)
        except Exception:
            # Fallback: try without branch spec (default branch)
            try:
                import git  # type: ignore[import]
                git.Repo.clone_from(repo_url, dest, depth=1)
            except Exception as e:
                raise BackendError(f"Failed to clone repository {repo_url}: {e}") from e

    def create_venv(
        self,
        venv_path: Path,
        python_version: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Create a UV virtual environment at *venv_path* (idempotent)."""
        if (venv_path / "bin" / "python").exists() or (venv_path / "Scripts" / "python.exe").exists():
            console.print(f"  [dim]Venv already exists at {venv_path}[/dim]")
            return

        console.print(f"  Creating venv at [cyan]{venv_path}[/cyan] ...")
        cmd = ["uv", "venv", str(venv_path)]
        if python_version:
            cmd = ["uv", "venv", "--python", python_version, str(venv_path)]

        if verbose:
            console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if verbose and result.stdout.strip():
            console.print(result.stdout.rstrip())
        if verbose and result.stderr.strip():
            console.print(f"  [dim]{result.stderr.rstrip()}[/dim]")

        if result.returncode != 0:
            raise BackendError(
                f"Failed to create venv: {result.stderr}\n"
                "Make sure `uv` is installed: https://github.com/astral-sh/uv"
            )

    def install_deps(
        self,
        repo_path: Path,
        venv_path: Path,
        extras: list[str] = [],
        verbose: bool = False,
    ) -> None:
        """Install ALL dependencies found in the repo into the venv.

        Installs in order:
          1. Every requirements*.txt / requirements/*.txt found
          2. The package itself via ``uv pip install .`` if setup.py or pyproject.toml exists
          3. Any extra requirements files declared in the recipe
        """
        python_bin = venv_path / "bin" / "python"
        if not python_bin.exists():
            python_bin = venv_path / "Scripts" / "python.exe"

        req_files = self._find_all_requirements(repo_path)
        has_installable = (
            (repo_path / "setup.py").exists()
            or (repo_path / "pyproject.toml").exists()
        )

        if not req_files and not has_installable and not extras:
            console.print("  [yellow]No requirements found — skipping dep install[/yellow]")
            return

        def _run(cmd: list[str], label: str) -> None:
            if verbose:
                console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")
            if verbose:
                result = subprocess.run(cmd, cwd=str(repo_path), text=True)
            else:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=str(repo_path)
                )
                if result.stdout.strip():
                    lines = result.stdout.strip().splitlines()
                    console.print(f"  [dim]{lines[-1]}[/dim]")
            if result.returncode != 0:
                raise BackendError(
                    f"Dependency installation failed ({label}):\n"
                    + (result.stderr or "")
                )

        # 1. Install every requirements file
        for req_file in req_files:
            console.print(f"  Installing deps from [cyan]{req_file.name}[/cyan] ...")
            _run(
                ["uv", "pip", "install", "--python", str(python_bin), "-r", str(req_file)],
                req_file.name,
            )

        # 2. Install the package itself if it is pip-installable
        if has_installable:
            console.print("  Installing package ([cyan]setup.py / pyproject.toml[/cyan]) ...")
            _run(
                ["uv", "pip", "install", "--python", str(python_bin), "-e", str(repo_path)],
                "package install",
            )

        # 3. Extra requirements files declared in the recipe
        for extra in extras:
            extra_path = repo_path / extra
            if extra_path.exists():
                console.print(f"  Installing extra deps from [cyan]{extra}[/cyan] ...")
                _run(
                    ["uv", "pip", "install", "--python", str(python_bin), "-r", str(extra_path)],
                    extra,
                )

    @staticmethod
    def _find_all_requirements(repo_path: Path) -> list[Path]:
        """Return every requirements file present in the repository."""
        # Explicit candidates checked first (in priority order)
        candidates = [
            "requirements.txt",
            "requirements-base.txt",
            "requirements-core.txt",
            "requirements-torch.txt",
            "requirements_torch.txt",
            "requirements-gpu.txt",
            "requirements-cuda.txt",
            "requirements/base.txt",
            "requirements/main.txt",
            "requirements/requirements.txt",
        ]
        found: list[Path] = []
        seen: set[Path] = set()

        for name in candidates:
            p = repo_path / name
            if p.exists() and p not in seen:
                found.append(p)
                seen.add(p)

        # Also pick up any other requirements*.txt at the repo root not yet covered
        for p in sorted(repo_path.glob("requirements*.txt")):
            if p not in seen:
                found.append(p)
                seen.add(p)

        return found


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Downloads and verifies model weights."""

    def fetch_hf(self, repo_id: str, dest: Path, token: str | None = None) -> None:
        """Download HuggingFace model via snapshot_download."""
        if dest.exists() and any(dest.iterdir()):
            console.print(f"  [dim]Model already at {dest}[/dim]")
            return

        dest.mkdir(parents=True, exist_ok=True)
        console.print(f"  Downloading HuggingFace model [cyan]{repo_id}[/cyan] ...")
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import]
            snapshot_download(repo_id, local_dir=str(dest), token=token)
        except Exception as e:
            raise ModelDownloadError(f"Failed to download {repo_id}: {e}") from e

    def fetch_url(self, url: str, dest: Path) -> None:
        """Download a file from *url* to *dest* with resume support."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            console.print(f"  [dim]File already at {dest}[/dim]")
            return

        console.print(f"  Downloading [cyan]{url}[/cyan] ...")
        try:
            import httpx  # type: ignore[import]

            headers: dict[str, str] = {}
            resume_pos = 0
            if dest.exists():
                resume_pos = dest.stat().st_size
                headers["Range"] = f"bytes={resume_pos}-"

            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + resume_pos

                with Progress(
                    TextColumn("[cyan]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(dest.name, total=total)
                    progress.advance(task, resume_pos)

                    mode = "ab" if resume_pos else "wb"
                    with open(dest, mode) as f:
                        for chunk in r.iter_bytes(chunk_size=65536):
                            f.write(chunk)
                            progress.advance(task, len(chunk))
        except Exception as e:
            raise ModelDownloadError(f"Download failed: {e}") from e

    def fetch_civitai(self, model_id: str, dest: Path, api_key: str | None = None) -> None:
        """Download a model from CIVITAI."""
        import httpx  # type: ignore[import]

        url = f"https://civitai.com/api/download/models/{model_id}"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        dest.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"  Downloading CIVITAI model [cyan]{model_id}[/cyan] ...")
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes(65536):
                        f.write(chunk)
        except Exception as e:
            raise ModelDownloadError(f"CIVITAI download failed: {e}") from e

    def verify(self, path: Path, expected_sha256: str) -> bool:
        """Return True if *path* has the expected SHA-256 checksum."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest().lower() == expected_sha256.lower()

    def download_model(self, model_desc: ModelDescriptor, models_dir: Path) -> Path:
        """Route to the correct fetcher and return local model path."""
        dest = models_dir / model_desc.destination
        src = model_desc.source
        if src == "huggingface":
            self.fetch_hf(model_desc.id, dest)
        elif src == "url":
            file_name = model_desc.id.split("/")[-1]
            self.fetch_url(model_desc.id, dest / file_name)
        elif src == "civitai":
            self.fetch_civitai(model_desc.id, dest / f"{model_desc.name}.safetensors")
        elif src == "local":
            if not dest.exists():
                raise ModelDownloadError(
                    f"Local model source does not exist: {dest}"
                )
        else:
            raise ModelDownloadError(f"Unknown model source: {src!r}")

        if model_desc.checksum and dest.exists():
            target = dest if dest.is_file() else next(dest.iterdir(), dest)
            if target.is_file() and not self.verify(target, model_desc.checksum):
                raise ModelDownloadError(
                    f"SHA-256 checksum mismatch for {model_desc.name}. "
                    "The file may be corrupt — delete it and retry."
                )

        return dest


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

class InferenceRunner:
    """Builds and executes inference subprocesses."""

    def build_command(
        self,
        recipe: Recipe,
        inputs: dict[str, Any],
        venv_path: Path,
        repo_path: Path,
        script_override: Path | None = None,
    ) -> list[str]:
        """Map recipe inputs to a CLI command list."""
        python_bin = venv_path / "bin" / "python"
        if not python_bin.exists():
            python_bin = venv_path / "Scripts" / "python.exe"

        script = script_override or (repo_path / recipe.entrypoint.script)

        if recipe.entrypoint.args_template:
            # Fill in template string
            filled = recipe.entrypoint.args_template.format(**inputs)
            return [str(python_bin), str(script)] + filled.split()

        cmd = [str(python_bin), str(script)]
        for key, value in inputs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])
        return cmd

    def execute(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Run *cmd* in *cwd* and return (returncode, stdout, stderr)."""
        import os
        run_env = {**os.environ, **(env or {})}
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=run_env,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr

    def collect_output(
        self,
        recipe: Recipe,
        stdout: str,
        cwd: Path,
        run_start: float = 0.0,
    ) -> dict[str, str]:
        """Collect outputs from file globs or stdout.

        Falls back to searching for files created after *run_start* when a
        recipe's output path pattern doesn't match the actual output location.
        """
        import datetime
        import glob as glob_mod

        outputs: dict[str, str] = {}
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for out_spec in recipe.outputs:
            if out_spec.type == "string":
                outputs[out_spec.name] = stdout.strip()
            elif out_spec.type == "file" and out_spec.path:
                pattern = out_spec.path.replace("{timestamp}", "*")
                matches = sorted(glob_mod.glob(str(cwd / pattern)))
                if matches:
                    outputs[out_spec.name] = matches[-1]
                else:
                    # Pattern missed — look for any recently created output file
                    recent = self._find_recent_files(cwd, run_start)
                    if recent:
                        outputs[out_spec.name] = recent[-1]
                    else:
                        # Return expected path so caller knows where to look
                        outputs[out_spec.name] = str(
                            cwd / out_spec.path.replace("{timestamp}", ts)
                        )
            else:
                outputs[out_spec.name] = stdout.strip()

        if not outputs:
            # No output spec — check for any new files produced
            recent = self._find_recent_files(cwd, run_start)
            if recent:
                outputs["result"] = recent[-1]
            else:
                outputs["stdout"] = stdout.strip()

        return outputs

    _OUTPUT_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
        ".mp4", ".avi", ".mov", ".mkv",
        ".wav", ".mp3", ".flac", ".ogg",
        ".json", ".txt",
    }

    def _find_recent_files(self, search_root: Path, since: float) -> list[str]:
        """Return paths of files under *search_root* created after *since* (epoch seconds)."""
        found: list[str] = []
        for p in search_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in self._OUTPUT_EXTENSIONS:
                continue
            try:
                if p.stat().st_mtime >= since:
                    found.append(str(p))
            except OSError:
                pass
        return sorted(found)


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------

class LocalBackend(AbstractBackend):
    """Local GPU/CPU compute backend (Phase 1)."""

    name = "local"

    def __init__(self, cache_dir: Path | None = None, verbose: bool = False):
        self.cache_dir = cache_dir or (Path.home() / ".kdream" / "cache")
        self.verbose = verbose
        self.env_manager = EnvironmentManager()
        self.model_manager = ModelManager()
        self.hardware = HardwareDetector()
        self.runner = InferenceRunner()

    def install(
        self,
        recipe: Recipe,
        cache_dir: Path,
        force_reinstall: bool = False,
    ):
        """Clone repo, create venv, install deps, download models."""
        from kdream.core.runner import PackageInfo

        pkg_dir = cache_dir / recipe.metadata.name
        repo_path = pkg_dir / "repo"
        venv_path = pkg_dir / "venv"
        models_path = pkg_dir / "models"

        pkg_dir.mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)

        if force_reinstall and pkg_dir.exists():
            import shutil
            shutil.rmtree(pkg_dir)
            pkg_dir.mkdir(parents=True, exist_ok=True)
            models_path.mkdir(parents=True, exist_ok=True)

        # Check GPU requirements
        hw = self.hardware.detect()
        local_spec = recipe.backends.local
        if local_spec and local_spec.requires_gpu and hw["device"] == "cpu":
            console.print(
                "[yellow]Warning:[/yellow] This recipe requires a GPU "
                "but none was detected. Inference may be very slow."
            )

        if self.verbose:
            console.print(f"  [dim]Hardware: {hw['device']}"
                          + (f", {hw['vram_gb']} GB VRAM" if hw['vram_gb'] else "")
                          + f" | cache: {cache_dir}[/dim]")

        # 1. Clone repo
        console.print(f"\n[bold][1/4][/bold] Cloning repo")
        self.env_manager.clone_repo(recipe.source.repo, recipe.source.ref, repo_path)

        # 2. Create venv
        console.print(f"\n[bold][2/4][/bold] Creating virtual environment")
        self.env_manager.create_venv(venv_path, verbose=self.verbose)

        # 3. Install dependencies
        console.print(f"\n[bold][3/4][/bold] Installing dependencies")
        self.env_manager.install_deps(
            repo_path, venv_path, recipe.source.install_extras, verbose=self.verbose
        )

        # 4. Download models
        console.print(f"\n[bold][4/4][/bold] Downloading models"
                      + (f" ({len(recipe.models)})" if recipe.models else ""))
        if recipe.models:
            for model in recipe.models:
                self.model_manager.download_model(model, models_path)
        else:
            console.print("  [dim]No models to download[/dim]")

        ready = repo_path.exists() and venv_path.exists()
        return PackageInfo(
            recipe_name=recipe.metadata.name,
            path=pkg_dir,
            ready=ready,
            venv_path=venv_path,
            repo_path=repo_path,
            models_path=models_path,
        )

    def run(self, package, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load the recipe, validate inputs, build command, execute, collect outputs."""
        # Re-load the recipe from the repo if stored, otherwise from registry
        recipe_file = package.repo_path / "kdream-recipe.yaml"
        if recipe_file.exists():
            from kdream.core.recipe import load_recipe
            recipe = load_recipe(str(recipe_file))
        else:
            # Build a minimal recipe for command construction
            from kdream.core.registry import RegistryClient
            try:
                recipe = RegistryClient().fetch_recipe(package.recipe_name)
            except Exception:
                raise BackendError(
                    f"Could not load recipe for '{package.recipe_name}'. "
                    "Re-install with force_reinstall=True."
                )

        errors = self.validate_inputs(recipe, inputs)
        if errors:
            raise BackendError(
                "Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        # Apply defaults for missing optional inputs
        merged: dict[str, Any] = {}
        for name, spec in recipe.inputs.items():
            if name in inputs:
                merged[name] = inputs[name]
            elif spec.default is not None:
                merged[name] = spec.default

        wrapper = self._ensure_cli_wrapper(recipe, package)
        cmd = self.runner.build_command(
            recipe, merged, package.venv_path, package.repo_path,
            script_override=wrapper,
        )
        if self.verbose:
            console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")
        else:
            console.print(f"  Running: [dim]{' '.join(cmd[:4])}…[/dim]")

        import os
        import shutil
        import time

        run_start = time.time()
        rc, stdout, stderr = self.runner.execute(cmd, cwd=package.repo_path)

        if self.verbose:
            if stdout.strip():
                console.print("[dim]--- stdout ---[/dim]")
                console.print(stdout.rstrip())
            if stderr.strip():
                console.print("[dim]--- stderr ---[/dim]")
                console.print(stderr.rstrip())

        if rc != 0:
            # Always show stderr on failure, even in non-verbose mode
            console.print(f"[dim]--- stderr ---[/dim]")
            console.print(stderr[-3000:].rstrip())
            raise BackendError(f"Inference failed (exit code {rc}).")

        raw_outputs = self.runner.collect_output(
            recipe, stdout, package.repo_path, run_start
        )

        # Copy file outputs to the directory where the user ran kdream
        dest_dir = Path(inputs.get("output_dir", os.getcwd()))
        dest_dir.mkdir(parents=True, exist_ok=True)

        final_outputs: dict[str, Any] = {}
        for name, value in raw_outputs.items():
            src = Path(value)
            if src.exists() and src.is_file():
                dest = dest_dir / src.name
                if src.resolve() != dest.resolve():
                    shutil.copy2(str(src), str(dest))
                final_outputs[name] = str(dest)
                console.print(f"  Saved: [cyan]{dest}[/cyan]")
            else:
                final_outputs[name] = value

        return final_outputs

    # ------------------------------------------------------------------
    # CLI wrapper generation
    # ------------------------------------------------------------------

    def _ensure_cli_wrapper(self, recipe: Recipe, package) -> Path | None:
        """Return a wrapper script path when the inference script has no CLI
        argument support, otherwise return None (use the original script).

        The wrapper is generated once at run time and cached in the repo dir.
        """
        script_path = package.repo_path / recipe.entrypoint.script
        if not script_path.exists():
            return None

        try:
            source = script_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        # If the script already handles CLI args, no wrapper needed
        cli_indicators = ["argparse", "click.command", "@app.command",
                          "fire.Fire", "sys.argv[", "typer"]
        if any(ind in source for ind in cli_indicators):
            return None

        wrapper_path = package.repo_path / "_kdream_runner.py"
        console.print("  [dim]Patching inference script to accept CLI inputs…[/dim]")
        wrapper_source = self._build_cli_wrapper(recipe, script_path, source)
        wrapper_path.write_text(wrapper_source, encoding="utf-8")

        return wrapper_path

    def _build_cli_wrapper(
        self, recipe: Recipe, script_path: Path, source: str
    ) -> str:
        """Generate a wrapper that:
        1. Parses recipe inputs as CLI flags via argparse.
        2. Uses Python AST to find top-level variable assignments matching
           recipe input names and injects override lines immediately after
           each one, so CLI values win over hardcoded defaults.
        """
        import ast

        # ── argparse declarations ────────────────────────────────────────
        arg_lines: list[str] = []
        for name, spec in recipe.inputs.items():
            flag = name.replace("_", "-")
            if spec.type == "boolean":
                arg_lines.append(
                    f'_kp.add_argument("--{flag}", action="store_true", default=False)'
                )
            elif spec.type == "integer":
                arg_lines.append(f'_kp.add_argument("--{flag}", type=int, default=None)')
            elif spec.type == "float":
                arg_lines.append(f'_kp.add_argument("--{flag}", type=float, default=None)')
            else:
                arg_lines.append(f'_kp.add_argument("--{flag}", type=str, default=None)')

        # ── AST scan: find top-level assignments matching input names ────
        input_names = set(recipe.inputs.keys())
        # end_lineno → list of override statements to insert after that line
        overrides: dict[int, list[str]] = {}
        try:
            tree = ast.parse(source)
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if not (isinstance(target, ast.Name) and target.id in input_names):
                        continue
                    varname = target.id
                    spec = recipe.inputs[varname]
                    cast = {"integer": "int", "float": "float", "boolean": "bool"}.get(
                        spec.type, "str"
                    )
                    override = (
                        f"if _kdream_args.get('{varname}') is not None: "
                        f"{varname} = {cast}(_kdream_args['{varname}'])"
                    )
                    if node.end_lineno is not None:
                        overrides.setdefault(node.end_lineno, []).append(override)
        except SyntaxError:
            pass

        # ── Inject override lines into source ────────────────────────────
        src_lines = source.splitlines()
        patched: list[str] = []
        for i, line in enumerate(src_lines, start=1):
            patched.append(line)
            if i in overrides:
                patched.extend(overrides[i])
        patched_source = "\n".join(patched)

        # ── Preamble ─────────────────────────────────────────────────────
        preamble_lines = [
            f'"""kdream CLI wrapper for {recipe.metadata.name} — auto-generated."""',
            "import argparse as _kap",
            "import os as _kos",
            f"_kos.chdir({repr(str(script_path.parent))})",
            f"__file__ = {repr(str(script_path))}",
            "_kp = _kap.ArgumentParser()",
            *arg_lines,
            "_kdream_args = {",
            '    k.replace("-", "_"): v',
            "    for k, v in vars(_kp.parse_known_args()[0]).items()",
            "    if v is not None",
            "}",
            "",
            "# ── Original script (patched with CLI overrides) ──",
            "",
        ]
        preamble = "\n".join(preamble_lines) + "\n"

        return preamble + patched_source

    def is_installed(self, recipe_name: str, cache_dir: Path) -> bool:
        pkg_dir = cache_dir / recipe_name
        return (pkg_dir / "repo").exists() and (pkg_dir / "venv").exists()
