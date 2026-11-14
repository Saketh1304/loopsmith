"""Small statistics helpers. One of these functions has a bug."""


def mean(numbers: list[float]) -> float:
    """Return the average of the numbers."""
    if not numbers:
        raise ValueError("mean() needs at least one number")
    return sum(numbers) / len(numbers)


def median(numbers: list[float]) -> float:
    """Return the middle value, or the average of the two middle values."""
    if not numbers:
        raise ValueError("median() needs at least one number")
    ordered = sorted(numbers)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle] + ordered[middle + 1]) / 2


def value_range(numbers: list[float]) -> float:
    """Return the difference between the largest and smallest number."""
    if not numbers:
        raise ValueError("value_range() needs at least one number")
    return max(numbers) - min(numbers)
