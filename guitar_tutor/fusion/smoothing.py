from __future__ import annotations


class Hysteresis:
    def __init__(self, high: float, low: float): self.high, self.low, self.value = high, low, False
    def update(self, score: float) -> bool:
        if not self.value and score >= self.high: self.value=True
        elif self.value and score < self.low: self.value=False
        return self.value
