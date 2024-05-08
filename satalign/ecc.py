"""
This module implements the eo-learn co-registration
(they use Enhanced Cross Correlation). The code has
been adapted from the `eo-learn` library. The original
code can be found at: 

https://github.com/sentinel-hub/eo-learn/blob/master/eolearn/coregistration/coregistration.py 

This source code is licensed under the MIT license.
"""

import warnings
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from satalign.main import SatAlign


class ECC(SatAlign):
    def __init__(
        self,
        datacube: np.ndarray,
        reference: np.ndarray,
        criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_COUNT, 100, 0),
        gauss_kernel_size: int = 3,
        **kwargs,
    ):
        """

        Args:
            datacube (xr.DataArray): The data cube to be aligned. The data cube
                needs to have the following dimensions: (time, bands, height, width).
            reference (Optional[xr.DataArray], optional): The reference image.
                The reference image needs to have the following dimensions:
                (bands, height, width).
            criteria (Tuple[int, int, float], optional): The strategy for the termination
                of the iterative search algorithm. The cv2.TERM_CRITERIA_COUNT indicates
                that the algorithm should terminate after a certain number of iterations.
            gauss_kernel_size (int, optional): The size of the Gaussian kernel used to
                smooth the images before calculating the warp matrix. Defaults to 1.
            **kwargs: Additional keyword arguments. See the `SatSync` class for more
                information.
        """
        super().__init__(datacube, reference, **kwargs)

        self.datacube = datacube
        self.reference = reference
        self.criteria = criteria
        self.gauss_kernel_size = gauss_kernel_size

    def find_warp(
        self,
        reference_image: np.ndarray,
        moving_image: np.ndarray,
    ) -> np.ndarray:
        """
        Find the warp matrix that aligns the source and
        destination image.

        Args:
            reference_image (numpy.ndarray): The source image
            moving_image (numpy.ndarray): The destination image

        Returns:
            numpy.ndarray: The aligned source image
        """

        try:
            _, warp_matrix = cv2.findTransformECC(
                templateImage=reference_image.astype(np.float32),
                inputImage=moving_image.astype(np.float32),
                warpMatrix=self.warp_matrix,
                motionType=cv2.MOTION_TRANSLATION,  # we only translate the images
                criteria=self.criteria,
                inputMask=None,
                gaussFiltSize=self.gauss_kernel_size,
            )

        except cv2.error as cv2err:
            warnings.warn(f"Could not calculate the warp matrix: {cv2err}")
            warp_matrix = np.eye(*self.warp_matrix_size, dtype=np.float32)
        return warp_matrix
