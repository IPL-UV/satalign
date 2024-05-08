import concurrent.futures
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import xarray as xr

class SatAlign(ABC):
    """
    An abstract class for multi-temporal image co-registration

    The task uses a temporal stack of images of the same location
    and a reference timeless image to estimate a translation
    matrix that aligns each frame of the temporal datacube to the
    reference image.

    Each transformation is calculated using only a single channel
    of the images. The estimated transformations are applied to
    each of the specified features.
    """

    def __init__(
        self,
        datacube: Union[xr.DataArray, np.ndarray],
        reference: Union[xr.DataArray, np.ndarray],
        channel: Union[int, str] = "mean",
        interpolation: int = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
        crop_center: Optional[int] = None,
        rgb_bands: List[int] = [3, 2, 1],
        border_mode: int = cv2.BORDER_REPLICATE,
        num_threads: int = 4,
        max_translations: int = 5.0,
        border_value: int = 0,
    ):
        """

        Args:
            datacube (xr.DataArray): The data cube to be aligned. The data cube
                needs to have the following dimensions: (time, bands, height, width).
            reference (Optional[xr.DataArray], optional): The reference image.
                The reference image needs to have the following dimensions:
                (bands, height, width).
            channel (Union[int, str], optional): The channel or feature to be used for
                alignment. Defaults to "gradients". The options are:
                - "gradients": The gradients of the image. It uses the Sobel operator
                    to calculate the gradients.
                - "mean": The mean of all the bands.
                - "luminance": The luminance of the image. It uses the following
                    formula: 0.299 * R + 0.587 * G + 0.114 * B.
                - int: The index of the band to be used.
            interpolation (int, optional): Interpolation type used when transforming
                the stack of images. Defaults to cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS.
            crop_center (Optional[int], optional): If this parameter is set, the
                images will be cropped with respect to the center of the image to
                calculate the warp matrix. The resulted warp matrix will be applied
                to the original image size. This can be useful for large images
                increasing the speed of the algorithm. Defaults to None, which means
                no cropping.
            rgb_bands (Tuple[int, int, int], optional): The RGB bands to be used to estimate
                the image features when the channel is set to "gradients" or "luminance".
                Defaults to [3, 2, 1].
            border_mode (int, optional): Defines the padding strategy when the warp matrix
                affects the border of the image. Defaults to cv2.BORDER_REPLICATE.
            num_threads (int, optional): Number of threads used to estimate the warp matrix.
                Only used in run_multicore method. The only method that supports multiple
                threads in a safe way is PCC. If ECC needs to be used, the cv2
                package have to be compiled with the flag WITH_TBB. Defaults to 4.
            max_translations (int, optional): Estimated transformations are considered
                incorrect when the norm of the translation component is larger than
                this parameter. Defaults to 5.0.
        """

        # Set the class attributes (REQUIRED)
        self.datacube = datacube
        self.reference = reference

        # Set the class attributes (OPTIONAL)
        self.channel = channel
        self.interpolation = interpolation
        self.rgb_bands = rgb_bands
        self.border_mode = border_mode
        self.crop_center = crop_center
        self.num_threads = num_threads
        self.max_translations = max_translations
        self.border_value = border_value

        # We do no support homography for now (AUTOMATIC)
        self.warp_matrix_size = (2, 3)
        self.warp_matrix: np.ndarray = np.eye(*self.warp_matrix_size, dtype=np.float32)

    @abstractmethod
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
        pass

    def warp_feature(self, img: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
        """
        Function to warp input image given an estimated
        affine transformation matrix.

        Args:
            img (numpy.ndarray): The input image

        Returns:
            numpy.ndarray: The warped image
        """

        height, width = img.shape[-2:]
        warped_img = np.zeros_like(img, dtype=np.float32)

        # Apply the transformation to each image
        for idx in range(img.shape[0]):
            warped_img[idx, ...] = cv2.warpAffine(
                src=img[idx, ...].astype(np.float32),
                M=warp_matrix,
                dsize=(width, height),
                borderMode=self.border_mode,
                flags=self.interpolation,
                borderValue=self.border_value,
            )

        return warped_img.astype(img.dtype)

    def is_translation_large(self, warp_matrix: np.ndarray) -> bool:
        """Method that checks if estimated linear translation
        could be implausible.

        This function checks whether the norm of the estimated
        translation in pixels exceeds a predefined value.

        Args:
            warp_matrix (numpy.ndarray): The estimated warp matrix
        """
        idist = np.linalg.norm(warp_matrix[:, 2]).astype(np.float32)
        return idist > self.max_translations

    def create_layer(self, img: np.ndarray) -> np.ndarray:
        """
        Method that generate a feature image from the input image

        Args:
            img (numpy.ndarray): The input image

        Returns:
            numpy.ndarray: The feature image
        """

        # If the image has more than 3 bands, select the RGB bands
        C, H, W = img.shape

        if C > 3:
            layer = img[self.rgb_bands]
        else:
            layer = img

        # Crop the image with respect to the centroid
        if self.crop_center is not None:

            if self.crop_center > H or self.crop_center > W:
                raise ValueError("The crop_center should be less than the image size")

            radius_x = (img.shape[-1] - self.crop_center) // 2
            radius_y = (img.shape[-2] - self.crop_center) // 2
            layer = layer[:, radius_y:-radius_y, radius_x:-radius_x]

        # From RGB to grayscale (image feature)
        if isinstance(self.channel, str):
            if self.channel == "gradients":
                global_reference = cv2.Sobel(
                    layer.mean(0).astype(np.float32), cv2.CV_32F, 1, 1
                )
            elif self.channel == "mean":
                global_reference = layer.mean(0)
            elif self.channel == "luminance":
                global_reference = (
                    layer[0] * 0.299 + layer[1] * 0.587 + layer[2] * 0.114
                )
        elif isinstance(self.channel, int):
            global_reference = layer[self.channel].copy()
        else:
            raise ValueError(
                "The channel should be a string (a specific method) or an integer (a band index)"
            )

        return global_reference

    def get_warped_image(
        self, reference_image_feature: np.ndarray, moving_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp the moving image to the reference image, using the
        estimated warp matrix.


        Args:
            reference_image (np.ndarray): The reference image
            moving_image (np.ndarray): The moving image

        Returns:
            Tuple[np.ndarray, np.ndarray]: The warped image and the
            estimated warp matrix
        """

        # Obtain the warp matrix
        warp_matrix = self.find_warp(
            reference_image=reference_image_feature,
            moving_image=self.create_layer(moving_image),
        )

        # Check if the translation is large
        if self.is_translation_large(warp_matrix):
            warnings.warn("Estimated translation is too large")
            warp_matrix = self.warp_matrix

        # Warp the image using the estimated warp matrix
        warped_image = self.warp_feature(img=moving_image, warp_matrix=warp_matrix)

        return warped_image, warp_matrix

    def run_xarray(self) -> xr.Dataset:
        """
        Run sequantially the get_warped_image method; input is xarray
        """
        # Create the reference feature using the reference image
        reference_layer = self.create_layer(self.reference.values)

        # Run iteratively the get_warped_image method
        warp_matrices = []
        warped_cube = np.zeros_like(self.datacube.values, dtype=self.datacube.dtype)
        for index, img in enumerate(self.datacube.values):
            # Obtain the warp matrix
            warped_image, warp_matrix = self.get_warped_image(
                reference_image_feature=reference_layer,
                moving_image=img,
            )

            # Save the warp matrix ... copy here makes cv2 happy
            # weird bug that makes in the final list have the same values
            # TODO: check in the future
            warp_matrices.append(warp_matrix.copy())
            warped_cube[index] = warped_image.copy()

        # Create the xarray dataset
        return xr.DataArray(
            data=warped_cube,
            coords=self.datacube.coords,
            dims=self.datacube.dims,
            attrs=self.datacube.attrs,
        ), warp_matrices

    def run_numpy(self) -> np.ndarray:
        """
        Run sequantially the get_warped_image method; input is numpy
        """

        # Create the reference feature using the reference image
        reference_layer = self.create_layer(self.reference)

        # Run iteratively the get_warped_image method
        warp_matrices = []
        warped_cube = np.zeros_like(self.datacube, dtype=self.datacube.dtype)
        for index, img in enumerate(self.datacube):
            # Obtain the warp matrix
            warped_image, warp_matrix = self.get_warped_image(
                reference_image_feature=reference_layer, moving_image=img
            )

            # Save the warp matrix ... copy here makes cv2 happy
            # weird bug that makes in the final list have the same values
            # TODO: check in the future
            warp_matrices.append(warp_matrix.copy())
            warped_cube[index] = warped_image.copy()

        return warped_cube, warp_matrices

    def run_multicore_numpy(self) -> np.ndarray:
        """
        Run the get_warped_image method using multiple threads
        """

        # Create the reference feature using the reference image
        reference_layer = self.create_layer(self.reference)

        # Create the executor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_threads
        ) as executor:

            futures = []
            for index, img in enumerate(self.datacube):
                futures.append(
                    executor.submit(
                        self.get_warped_image,
                        reference_image_feature=reference_layer,
                        moving_image=img,
                    )
                )

            # Save the results in the final list
            warped_cube = np.zeros_like(self.datacube, dtype=self.datacube.dtype)
            warp_matrices = []
            for index, future in enumerate(futures):
                warped_image, warp_matrix = future.result()
                warped_cube[index] = warped_image
                warp_matrices.append(warp_matrix)

        return warped_cube, warp_matrices

    def run_multicore_xarray(self) -> xr.Dataset:
        """
        Run the get_warped_image method using multiple threads
        """

        # Create the reference feature using the reference image
        reference_layer = self.create_layer(self.reference.values)

        # Create the executor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_threads
        ) as executor:

            futures = []
            for index, img in enumerate(self.datacube.values):
                futures.append(
                    executor.submit(
                        self.get_warped_image,
                        reference_image_feature=reference_layer,
                        moving_image=img
                    )
                )

            # Save the results in the final list
            warped_cube = np.zeros_like(self.datacube.values, dtype=self.datacube.dtype)
            warp_matrices = []
            for index, future in enumerate(futures):
                warped_image, warp_matrix = future.result()
                warped_cube[index] = warped_image
                warp_matrices.append(warp_matrix)

        # Create the xarray dataset
        return xr.DataArray(
            data=warped_cube,
            coords=self.datacube.coords,
            dims=self.datacube.dims,
            attrs=self.datacube.attrs,
        ), warp_matrices

    def run(self) -> Union[xr.Dataset, np.ndarray]:
        """
        Run the alignment method
        """

        if isinstance(self.datacube, xr.DataArray):
            return self.run_xarray()
        else:
            return self.run_numpy()
        
    def run_multicore(self) -> Union[xr.Dataset, np.ndarray]:
        """
        Run the alignment method using multiple threads
        """

        if isinstance(self.datacube, xr.DataArray):
            return self.run_multicore_xarray()
        else:
            return self.run_multicore_numpy()