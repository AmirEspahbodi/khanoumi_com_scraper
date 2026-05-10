class BotChallengeDetected(RuntimeError):
    """Raised when a bot-challenge page is detected."""


class MaxRetriesExceeded(RuntimeError):
    """Raised when all retry attempts have been exhausted."""
