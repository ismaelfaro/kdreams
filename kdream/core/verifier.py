"""Recipe component verification — checks that all referenced resources exist
before a recipe is saved or used."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kdream.exceptions import RecipeError


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ComponentIssue:
    """A single verification finding."""
    severity: Literal["error", "warning"]
    component: str   # e.g. "model:flux-weights", "entrypoint", "source-repo"
    message: str

    def __str__(self) -> str:
        icon = "✗" if self.severity == "error" else "⚠"
        return f"{icon} [{self.component}] {self.message}"


@dataclass
class VerificationResult:
    """Aggregated result of verifying all recipe components."""
    issues: list[ComponentIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ComponentIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ComponentIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True only when there are no errors (warnings are allowed)."""
        return len(self.errors) == 0

    def raise_if_errors(self) -> None:
        """Raise :class:`~kdream.exceptions.RecipeError` if any errors exist."""
        if self.errors:
            lines = ["Recipe verification failed — missing or unreachable components:"]
            for issue in self.errors:
                lines.append(f"  {issue}")
            if self.warnings:
                lines.append("Warnings:")
                for issue in self.warnings:
                    lines.append(f"  {issue}")
            raise RecipeError("\n".join(lines))


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class RecipeVerifier:
    """Verify that all components referenced in a recipe actually exist.

    Checks performed (in order):

    1. **Inference implementation** — the entrypoint script exists in the
       source repository (GitHub raw-content HEAD request), or a generated
       runner script is present (HuggingFace / hybrid recipes).

    2. **Models** — each model descriptor is reachable:
       - ``huggingface``: model ID exists on the HF Hub (``api.model_info``).
       - ``url``: HTTP HEAD request succeeds.
       - ``civitai``: CIVITAI download endpoint responds (HEAD).
       - ``local``: emits a *warning* (path is on the user's machine and cannot
         be verified at generation time).

    3. **Source repository** — the source repo URL is reachable (HEAD).

    All network calls use a short timeout (8 s) so the verifier fails fast
    rather than hanging indefinitely.
    """

    TIMEOUT = 8  # seconds

    def verify(
        self,
        recipe,          # kdream.core.recipe.Recipe
        runner_script: str | None = None,
    ) -> VerificationResult:
        """Run all checks and return a :class:`VerificationResult`.

        Args:
            recipe:        The recipe to verify.
            runner_script: Content of a generated companion runner script
                           (set on HF / hybrid recipes; bypasses the
                           entrypoint-in-repo check).
        """
        result = VerificationResult()

        result.issues += self._check_entrypoint(recipe, runner_script)
        result.issues += self._check_models(recipe)
        result.issues += self._check_source_repo(recipe)

        return result

    # ------------------------------------------------------------------
    # Entrypoint / inference implementation
    # ------------------------------------------------------------------

    def _check_entrypoint(self, recipe, runner_script: str | None) -> list[ComponentIssue]:
        issues: list[ComponentIssue] = []
        script = (recipe.entrypoint.script or "").strip()

        if not script:
            issues.append(ComponentIssue(
                severity="error",
                component="entrypoint",
                message="No entrypoint script defined in the recipe.",
            ))
            return issues

        # If a generated runner script is present the entrypoint is satisfied.
        if runner_script:
            return issues

        # Try to verify the script exists in the GitHub repo.
        repo_url = (recipe.source.repo or "").strip()
        if not repo_url:
            issues.append(ComponentIssue(
                severity="error",
                component="entrypoint",
                message=(
                    f"Cannot verify entrypoint script '{script}': "
                    "source.repo is empty."
                ),
            ))
            return issues

        ref = (recipe.source.ref or "main").strip()
        raw_url = self._github_raw_url(repo_url, ref, script)
        if raw_url:
            reachable, detail = self._head(raw_url)
            if not reachable:
                issues.append(ComponentIssue(
                    severity="error",
                    component="entrypoint",
                    message=(
                        f"Inference script '{script}' not found in repository "
                        f"(checked: {raw_url}). {detail}"
                    ),
                ))
        else:
            # Non-GitHub source — emit a warning rather than an error
            issues.append(ComponentIssue(
                severity="warning",
                component="entrypoint",
                message=(
                    f"Cannot verify entrypoint script '{script}': "
                    f"source.repo is not a recognised GitHub URL ({repo_url!r}). "
                    "Skipping remote check."
                ),
            ))

        return issues

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def _check_models(self, recipe) -> list[ComponentIssue]:
        issues: list[ComponentIssue] = []

        if not recipe.models:
            issues.append(ComponentIssue(
                severity="warning",
                component="models",
                message=(
                    "No models declared in the recipe. "
                    "The inference script will be responsible for downloading "
                    "any required weights at runtime."
                ),
            ))
            return issues

        for model in recipe.models:
            component_id = f"model:{model.name}"
            src = model.source

            if src == "huggingface":
                ok, msg = self._check_hf_model(model.id)
                if not ok:
                    issues.append(ComponentIssue(
                        severity="error",
                        component=component_id,
                        message=msg,
                    ))

            elif src == "url":
                reachable, detail = self._head(model.id)
                if not reachable:
                    issues.append(ComponentIssue(
                        severity="error",
                        component=component_id,
                        message=(
                            f"Model URL unreachable: {model.id}. {detail}"
                        ),
                    ))

            elif src == "civitai":
                ok, msg = self._check_civitai(model.id)
                if not ok:
                    issues.append(ComponentIssue(
                        severity="warning",
                        component=component_id,
                        message=msg,
                    ))

            elif src == "local":
                issues.append(ComponentIssue(
                    severity="warning",
                    component=component_id,
                    message=(
                        f"Model '{model.name}' uses source=local. "
                        "Cannot verify the path at generation time — "
                        "ensure the file exists on the target machine."
                    ),
                ))

        return issues

    # ------------------------------------------------------------------
    # Source repository
    # ------------------------------------------------------------------

    def _check_source_repo(self, recipe) -> list[ComponentIssue]:
        issues: list[ComponentIssue] = []
        repo_url = (recipe.source.repo or "").strip()

        if not repo_url:
            issues.append(ComponentIssue(
                severity="error",
                component="source-repo",
                message="source.repo is empty — no repository to install from.",
            ))
            return issues

        # For HuggingFace model repos the "repo" field is the HF model URL,
        # which may 404 on a HEAD because HF uses redirects. Skip the HEAD
        # check for those and trust the HF model check already done above.
        if "huggingface.co" in repo_url:
            return issues

        reachable, detail = self._head(repo_url)
        if not reachable:
            issues.append(ComponentIssue(
                severity="warning",
                component="source-repo",
                message=(
                    f"Source repository may be unreachable: {repo_url}. "
                    f"{detail} "
                    "(This could be a private repo or a temporary network issue.)"
                ),
            ))

        return issues

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    def _head(self, url: str) -> tuple[bool, str]:
        """Issue an HTTP HEAD to *url*. Returns (reachable, detail_message)."""
        try:
            import httpx
            resp = httpx.head(url, follow_redirects=True, timeout=self.TIMEOUT)
            if resp.status_code < 400:
                return True, ""
            return False, f"HTTP {resp.status_code}"
        except Exception as exc:
            return False, str(exc)

    def _check_hf_model(self, model_id: str) -> tuple[bool, str]:
        """Return (exists, message) for a HuggingFace model ID."""
        try:
            from huggingface_hub import HfApi  # type: ignore[import]
            api = HfApi()
            api.model_info(model_id, timeout=self.TIMEOUT)
            return True, ""
        except Exception as exc:
            msg = str(exc)
            if "404" in msg or "not found" in msg.lower() or "repository" in msg.lower():
                return False, (
                    f"HuggingFace model '{model_id}' does not exist or is private. "
                    "Check the model ID and ensure you have access."
                )
            return False, (
                f"Could not verify HuggingFace model '{model_id}': {exc}. "
                "This may be a network issue — the model may still exist."
            )

    def _check_civitai(self, model_id: str) -> tuple[bool, str]:
        """Return (reachable, message) for a CIVITAI model ID."""
        url = f"https://civitai.com/api/download/models/{model_id}"
        reachable, detail = self._head(url)
        if not reachable:
            return False, (
                f"CIVITAI model '{model_id}' could not be verified: {detail}. "
                "The model may require an API key or may not exist."
            )
        return True, ""

    @staticmethod
    def _github_raw_url(repo_url: str, ref: str, script: str) -> str:
        """Convert a GitHub repo URL to a raw.githubusercontent.com URL for *script*.

        Returns an empty string if *repo_url* is not a recognised GitHub URL.
        """
        import re
        m = re.match(
            r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
            repo_url.strip(),
        )
        if not m:
            return ""
        org, repo = m.group(1), m.group(2)
        return f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{script}"


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def verify_recipe(recipe, runner_script: str | None = None) -> VerificationResult:
    """Verify a recipe's components and return a :class:`VerificationResult`.

    This is the main entry point for callers that don't need a verifier
    instance. It creates a :class:`RecipeVerifier`, runs all checks, and
    returns the result.

    Args:
        recipe:        The recipe to verify.
        runner_script: Generated companion runner script content (if any).

    Returns:
        :class:`VerificationResult` — inspect ``.ok``, ``.errors``,
        ``.warnings``, or call ``.raise_if_errors()`` to abort on problems.
    """
    return RecipeVerifier().verify(recipe, runner_script=runner_script)
