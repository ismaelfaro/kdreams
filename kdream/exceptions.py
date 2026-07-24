"""kdream exception hierarchy."""


class KdreamError(Exception):
    """Base exception for all kdream errors."""


class RecipeError(KdreamError):
    """Recipe parsing or validation error."""


class RegistryError(KdreamError):
    """Registry fetch or cache error."""


class BackendError(KdreamError):
    """Backend installation or execution error."""


class ModelDownloadError(KdreamError):
    """Model weight download or verification error."""
