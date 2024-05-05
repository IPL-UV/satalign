"""
This module implements the eo-learn co-registration (Enhanced Cross 
Correlation). The code has been adapted from the `eo-learn` library.
The original code can be found at: 

https://github.com/sentinel-hub/eo-learn/blob/master/eolearn/coregistration/coregistration.py 

This source code is licensed under the MIT license.
"""

import warnings
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from satsync.main import SuperAlignmentAbstract


class ECC(SuperAlignmentAbstract):
    """
    Multi-temporal image co-registration using OpenCV Enhanced
    Cross-Correlation method - eo-learn adaptation.
    """

    def __init__(
        self,
        datacube: np.ndarray,
        reference: np.ndarray,
        criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_COUNT, 100, 0),
        gauss_kernel_size: int = 1,
        warp_mode: int = cv2.MOTION_TRANSLATION,
        **kwargs,
    ):
        """Constructor for the ECC class"""
        super().__init__(datacube, reference, **kwargs)

        self.datacube = datacube
        self.reference = reference
        self.criteria = criteria
        self.gauss_kernel_size = gauss_kernel_size
        self.warp_mode = warp_mode

    def find_warp(self, reference_image, moving_image):
        try:
            _, warp_matrix = cv2.findTransformECC(
                templateImage=reference_image.astype(np.float32),
                inputImage=moving_image.astype(np.float32),
                warpMatrix=self.warp_matrix,
                motionType=self.warp_mode,
                criteria=self.criteria,
                inputMask=None,
                gaussFiltSize=self.gauss_kernel_size,
            )
        except cv2.error as cv2err:
            warnings.warn(f"Could not calculate the warp matrix: {cv2err}")
            warp_matrix = np.eye(*self.warp_matrix_size, dtype=np.float32)
        return warp_matrix
