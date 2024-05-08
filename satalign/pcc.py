import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from skimage.registration import phase_cross_correlation

from satalign.main import SatAlign


class PCC(SatAlign):
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
        """

        Args:
            datacube (xr.DataArray): The data cube to be aligned. The data cube
                needs to have the following dimensions: (time, bands, height, width).
            reference (Optional[xr.DataArray], optional): The reference image.
                The reference image needs to have the following dimensions:
                (bands, height, width).
            upsample_factor (Optional[int], optional): Upsampling factor. Images will
                be registered to within ``1 / upsample_factor`` of a pixel. For example
                ``upsample_factor == 20`` means the images will be registered within 1/20th
                of a pixel. Default is 1 (no upsampling).
            space (Optional[str], optional): One of "real" or "fourier". Defines how the
                algorithm interprets input data. "real" means data will be FFT'd to compute
                the correlation, while "fourier" data will bypass FFT of input data. Case
                insensitive.
            disambiguate (Optional[bool], optional): The shift returned by this function
                is only accurate *modulo* the image shape, due to the periodic nature of
                the Fourier transform. If this parameter is set to ``True``, the *real*
                space cross-correlation is computed for each possible shift, and the shift
                with the highest cross-correlation within the overlapping area is returned.
            overlap_ratio (Optional[float], optional): Minimum allowed overlap ratio between
                images. The correlation for translations corresponding with an overlap ratio
                lower than this threshold will be ignored. A lower `overlap_ratio` leads to
                smaller maximum translation, while a higher `overlap_ratio` leads to greater
                robustness against spurious matches due to small overlap between masked images.
        """
        super().__init__(datacube, reference, **kwargs)

        self.datacube = datacube
        self.reference = reference
        self.upsample_factor = upsample_factor
        self.space = space
        self.disambiguate = disambiguate
        self.overlap_ratio = overlap_ratio

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

        Raises:
            ValueError: No keypoints found in the moving image.
            ValueError: No matching points found.

        Returns:
            numpy.ndarray: The aligned source image
        """
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
