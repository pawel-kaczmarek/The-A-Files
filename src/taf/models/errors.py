class CapacityError(ValueError):
    """The message does not fit into the cover (or stego) signal.

    Methods raise this instead of a bare ``ValueError`` when the payload
    exceeds what the signal can structurally carry, so that an evaluation can
    record the trial as over capacity rather than as a crash or as a high bit
    error rate. It subclasses ``ValueError`` so existing callers that catch
    ``ValueError`` keep working.
    """
