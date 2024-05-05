"""
This module implements the eo-learn co-registration (Enhanced Cross 
Correlation). The code has been adapted from the `eo-learn` library.
The original code can be found at: 

https://github.com/sentinel-hub/eo-learn/blob/master/eolearn/coregistration/coregistration.py 

This source code is licensed under the MIT license.
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from skimage.registration import phase_cross_correlation

from satsync.main import SuperAlignmentAbstract


class PCC(SuperAlignmentAbstract):
    """
    Multi-temporal image co-registration using Phase Cross-Correlation
    """

    def __init__(
        self,
        datacube: np.ndarray,
        reference: np.ndarray,
        upsample_factor: Optional[int] = 50,
        space: Optional[str] = "real",
        disambiguate: Optional[bool] = False,
        overlap_ratio: Optional[float] = 0.3,
        **kwargs,
    ):
        """Constructor for the PCC class

        Args:
            datacube (np.ndarray): The data cube to be aligned.
            reference (np.ndarray): The reference feature.
            upsample_factor (Optional[int], optional): The upsample factor. Defaults to 1.
            space (Optional[str], optional): The space. Defaults to "real".
            disambiguate (Optional[bool], optional): The disambiguate. Defaults to False.
            overlap_ratio (Optional[float], optional): The overlap ratio. Defaults to 0.3.

        """
        super().__init__(datacube, reference, **kwargs)

        self.datacube = datacube
        self.reference = reference
        self.upsample_factor = upsample_factor
        self.space = space
        self.disambiguate = disambiguate
        self.overlap_ratio = overlap_ratio

    def find_warp(self, reference_image, moving_image):
        try:
            shift, error, diffphase = phase_cross_correlation(
                reference_image=reference_image,
                moving_image=moving_image,
                upsample_factor=self.upsample_factor,
                space=self.space,
                reference_mask=None,
                moving_mask=None,
                disambiguate=self.disambiguate,
                overlap_ratio=self.overlap_ratio,
            )

            # Create the warp matrix with the shift
            warp_matrix = np.eye(*self.warp_matrix_size, dtype=np.float32)
            warp_matrix[:2, 2] = shift[::-1]

        except Exception as err:
            warnings.warn(f"Could not calculate the warp matrix: {err}")
            warp_matrix = np.eye(*self.warp_matrix_size, dtype=np.float32)
        return warp_matrix
