"""HuggingFace Hub exploration — discover and browse the latest AI models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Task aliases: user-friendly name → HF pipeline tag
HF_TASK_ALIASES: dict[str, str] = {
    # Generative
    "image-generation":    "text-to-image",
    "text-to-image":       "text-to-image",
    "text-generation":     "text-generation",
    "chat":                "text-generation",
    "llm":                 "text-generation",
    "audio-generation":    "text-to-audio",
    "text-to-audio":       "text-to-audio",
    "text-to-speech":      "text-to-speech",
    "tts":                 "text-to-speech",
    "video-generation":    "text-to-video",
    "text-to-video":       "text-to-video",
    "image-to-image":      "image-to-image",
    "image-to-video":      "image-to-video",
    # Understanding
    "transcription":       "automatic-speech-recognition",
    "asr":                 "automatic-speech-recognition",
    "translation":         "translation",
    "summarization":       "summarization",
    "classification":      "text-classification",
    "image-classification":"image-classification",
    "object-detection":    "object-detection",
    "depth-estimation":    "depth-estimation",
    "embedding":           "feature-extraction",
}

# Canonical HF pipeline tags → friendly display name
_TASK_DISPLAY: dict[str, str] = {
    "text-to-image":               "image-generation",
    "text-generation":             "text-generation",
    "text-to-audio":               "audio-generation",
    "text-to-speech":              "text-to-speech",
    "text-to-video":               "video-generation",
    "image-to-image":              "image-to-image",
    "image-to-video":              "image-to-video",
    "automatic-speech-recognition":"transcription",
    "translation":                 "translation",
    "summarization":               "summarization",
    "text-classification":         "classification",
    "image-classification":        "image-classification",
    "object-detection":            "object-detection",
    "depth-estimation":            "depth-estimation",
    "feature-extraction":          "embedding",
}


@dataclass
class HFModel:
    """Lightweight representation of a HuggingFace model."""
    model_id: str
    author: str
    task: str                          # HF pipeline_tag
    tags: list[str] = field(default_factory=list)
    likes: int = 0
    downloads: int = 0
    description: str = ""
    license: str = ""
    last_modified: str = ""

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/{self.model_id}"

    @property
    def task_display(self) -> str:
        return _TASK_DISPLAY.get(self.task, self.task)

    def to_hf_repo_url(self) -> str:
        """Return the HuggingFace URL suitable for `kdream generate`."""
        return self.hf_url


def _resolve_task(task: str) -> str:
    """Normalise a user-supplied task string to an HF pipeline tag."""
    return HF_TASK_ALIASES.get(task.lower(), task.lower())


def search_hf_models(
    query: str | None = None,
    task: str | None = None,
    limit: int = 20,
    sort: str = "likes",
    author: str | None = None,
) -> list[HFModel]:
    """Search and browse models on the HuggingFace Hub.

    Args:
        query:  Free-text search query (model name, description keywords).
        task:   Task/pipeline filter, e.g. ``"text-to-image"``, ``"text-generation"``,
                ``"image-generation"``.  Friendly aliases are accepted
                (see :data:`HF_TASK_ALIASES`).
        limit:  Maximum number of results (default 20).
        sort:   Sort key — ``"likes"`` (default), ``"downloads"``, or
                ``"lastModified"``.
        author: Filter by HuggingFace author/organisation username.

    Returns:
        List of :class:`HFModel` objects sorted by *sort* descending.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from e

    hf_task = _resolve_task(task) if task else None

    api = HfApi()
    raw = api.list_models(
        search=query or None,
        task=hf_task,
        author=author or None,
        sort=sort,
        direction=-1,
        limit=limit,
        cardData=True,
        full=True,
    )

    models: list[HFModel] = []
    for m in raw:
        card = m.cardData or {}
        description = ""
        if isinstance(card, dict):
            description = str(card.get("model-index", [{"results": []}]) or "")
        # Try to get a license
        lic = ""
        if isinstance(card, dict):
            lic = card.get("license", "")
        tags = [t for t in (m.tags or []) if not t.startswith("license:") and len(t) < 40]
        # last modified date
        last_mod = ""
        if m.lastModified:
            try:
                last_mod = str(m.lastModified)[:10]
            except Exception:
                pass

        models.append(HFModel(
            model_id=m.modelId or "",
            author=m.author or (m.modelId or "").split("/")[0],
            task=m.pipeline_tag or "",
            tags=tags[:6],
            likes=m.likes or 0,
            downloads=m.downloads or 0,
            description=description[:120],
            license=str(lic) if lic else "",
            last_modified=last_mod,
        ))

    return models


def get_hf_model_readme(model_id: str) -> str:
    """Fetch the README/model card text for a HuggingFace model."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
        import os
        path = hf_hub_download(model_id, filename="README.md")
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""
