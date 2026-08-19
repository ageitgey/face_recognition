class Face2AIError(Exception):
    """Base domain error."""


class RecognitionUnavailable(Face2AIError):
    pass


class InvalidFrame(Face2AIError):
    pass


class EnrollmentRejected(Face2AIError):
    pass


class IdentityStoreCorrupted(Face2AIError):
    """Raised when persisted identity data cannot be safely decoded or validated."""


class IdentityStoreUnavailable(Face2AIError):
    """Raised when the identity store cannot be reached (an OSError), as opposed to being corrupt.

    The data may be perfectly valid and simply unreachable — a missing mount, a permission change,
    a full disk. Callers map this to HTTP 503; ``IdentityStoreCorrupted`` keeps its own meaning.
    """
