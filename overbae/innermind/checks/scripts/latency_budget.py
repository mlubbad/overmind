"""The llm call took at most MAX_SECONDS."""

MAX_SECONDS = None


def evaluate(span):
    duration = span.duration_s
    if MAX_SECONDS is None or isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return None
    return duration <= MAX_SECONDS, f"{round(duration, 3)}s, budget {MAX_SECONDS}s"
