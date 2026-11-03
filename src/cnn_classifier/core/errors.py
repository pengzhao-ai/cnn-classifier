"""Exception hierarchy for cnn-classifier."""


class CnnClassifierError(Exception):
    """Base class for all package-specific errors."""


class ConfigError(CnnClassifierError):
    """A configuration is malformed or incompatible with the selected device."""


class DataError(CnnClassifierError):
    """A dataset, image, manifest, or split is invalid."""


class RegistryError(CnnClassifierError):
    """A registry lookup or registration failed."""


class RunError(CnnClassifierError):
    """A run, checkpoint, or artifact is invalid."""
