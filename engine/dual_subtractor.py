"""Dual Background Subtractor for robust stationary object extraction."""

from typing import Tuple
import cv2
import numpy as np


class DualSubtractor:
    """Extracts stationary foreground targets using dual MOG2 subtractors."""

    def __init__(
        self,
        slow_history: int = 5000,
        slow_learning_rate: float = 0.0001,
        fast_history: int = 40,
        fast_learning_rate: float = 0.05,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
    ) -> None:
        self.slow_history: int = slow_history
        self.slow_learning_rate: float = slow_learning_rate
        self.fast_history: int = fast_history
        self.fast_learning_rate: float = fast_learning_rate

        self.subtractor_slow: cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=self.slow_history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.subtractor_fast: cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=self.fast_history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        self._kernel_open: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._kernel_close: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process frame and return (slow_mask, fast_mask, static_cleaned_mask)."""
        slow_mask = self.subtractor_slow.apply(frame, learningRate=self.slow_learning_rate)
        fast_mask = self.subtractor_fast.apply(frame, learningRate=self.fast_learning_rate)

        # Vectorized shadow stripping: pixel value 255 = true foreground, 127 = shadow
        slow_fg = slow_mask == 255
        fast_fg = fast_mask == 255

        # Static extraction: present in slow model, but absorbed by fast model
        static_raw = slow_fg & (~fast_fg)
        static_mask = (static_raw.astype(np.uint8)) * 255

        # Morphological noise removal & contour consolidation
        opened = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, self._kernel_open)
        static_clean = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel_close)

        return slow_mask, fast_mask, static_clean
