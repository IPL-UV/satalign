"""
An abstract class for multi-temporal image co-registration

This source code is licensed under the MIT license.
"""

import concurrent.futures
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class SuperAlignmentAbstract(ABC):
    """
    An abstract class for multi-temporal image co-registration

    The task uses a temporal stack of images of the same location
    and a reference timeless feature to estimate a transformation
    that aligns each frame of the temporal stack to the reference
    feature.

    Each transformation is calculated using only a single channel
    of the images. The estimated transformations are applied to
    each of the specified features.
    """

    def __init__(
        self,
        datacube: np.ndarray,
        reference: np.ndarray,
        channel: Union[int, str] = "luminance",
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
            datacube (xr.DataArray): The data cube to be aligned.
            reference (Optional[xr.DataArray], optional): The reference feature.
                Defaults to median of the images after 2022-01-01.
            channel (Union[int, str], optional): The channel to be used for
                alignment. Defaults to "gradients".
            interpolation (int, optional): Interpolation type used when transforming
                the stack of images. Defaults to cv2.INTER_LINEAR.
            crop_center (Optional[int], optional): If this parameter is set, the
                images will be cropped with respect to the center of the image to
                calculate the warp matrix. The affine transformation will be applied
                to the original image size. This can be useful for large images
                increasing the speed of the algorithm. Defaults to None, which means
                no cropping.
            rgb_bands (Tuple[int, int, int], optional): The RGB bands to be used
                when finding the gradient. Defaults to (3, 2, 1).
            border_mode (int, optional): Defines the padding strategy when transforming
                the images with estimated transformation. Defaults to cv2.BORDER_REPLICATE.
            num_threads (int, optional): Number of threads used to estimate the warp.
            max_translations (int, optional): Estimated transformations are considered
                incorrect when the norm of the translation component is larger than
                this parameter. Defaults to 5.0.
        """

        # Set the class attributes
        self.datacube = datacube
        self.reference = reference
        self.channel = channel
        self.interpolation = interpolation
        self.rgb_bands = rgb_bands
        self.border_mode = border_mode
        self.crop_center = crop_center
        self.num_threads = num_threads
        self.max_translations = max_translations
        self.border_value = border_value

        # Global reference
        self.warp_matrix_size = (2, 3)
        self.warp_matrix: np.ndarray = np.eye(*self.warp_matrix_size, dtype=np.float32)

    @abstractmethod
    def find_warp(self, reference_image, moving_image):
        """
        Find the transformation between the source
        and destination image.

        Args:
            reference_image (numpy.ndarray): The source image
            moving_image (numpy.ndarray): The destination image

        Returns:
            numpy.ndarray: The aligned source image
        """
        pass

    def warp(
        self,
        img: np.ndarray,
        warp_matrix: np.ndarray,
        shape: Tuple[int, int],
        flags: int,
    ) -> np.ndarray:
        """
        Method that applies the estimated transformation
        to the source image

        Args:
            img (numpy.ndarray): The source image
            warp_matrix (numpy.ndarray): The transformation matrix
            shape (Tuple[int, int]): The shape of the destination image
            flags (int): The flags for the transformation
        """

        return cv2.warpAffine(
            img.astype(np.float32),
            warp_matrix,
            shape,
            borderMode=self.border_mode,
            flags=flags,
            borderValue=self.border_value,
        )

    def warp_feature(self, img: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
        """
        Function to warp input image given an estimated
        2D linear transformation

        Args:
            img (numpy.ndarray): The input image
            warp_matrix (numpy.ndarray): The estimated
                transformation matrix
        """

        height, width = img.shape[-2:]
        warped_img = np.zeros_like(img, dtype=np.float32)

        # Apply the transformation to the image
        for idx in range(img.shape[0]):
            warped_img[idx, ...] = self.warp(
                img=img[idx, ...].astype(np.float32),
                warp_matrix=warp_matrix,
                shape=(width, height),
                flags=self.interpolation,
            )

        return warped_img.astype(img.dtype)

    def is_translation_large(self, warp_matrix: np.ndarray) -> bool:
        """Method that checks if estimated linear translation
        could be implausible.

        This function checks whether the norm of the estimated
        translation in pixels exceeds a predefined value.

        Args:
            warp_matrix (numpy.ndarray): The estimated transformation
                matrix
        """
        idist = np.linalg.norm(warp_matrix[:, 2]).astype(np.float32)
        return idist > self.max_translations

    def create_layer(self, img: np.ndarray) -> np.ndarray:
        """
        Method that creates a gradient image from the input image

        Args:
            img (numpy.ndarray): The input image
        """
        # If the image has more than 3 bands, select the RGB bands
        C, H, W = img.shape

        if C > 3:
            layer = img[self.rgb_bands].copy()
        else:
            layer = img.copy()

        # Crop the image with respect to the centroid
        if self.crop_center is not None:
            # if crop_center > to the original image
            if self.crop_center > H or self.crop_center > W:
                raise ValueError("The crop_center should be less than the image size")
            radius_x = (img.shape[-1] - self.crop_center) // 2
            radius_y = (img.shape[-2] - self.crop_center) // 2
            layer = layer[:, radius_y:-radius_y, radius_x:-radius_x]

        # From RGB to grayscale
        if self.channel == "gradients":
            global_reference = cv2.Sobel(
                layer.mean(0).astype(np.float32), cv2.CV_32F, 1, 1
            )
        elif self.channel == "mean":
            global_reference = layer.mean(0)
        elif self.channel == "luminance":
            global_reference = layer[0] * 0.299 + layer[1] * 0.587 + layer[2] * 0.114
        else:
            global_reference = self.reference[self.channel]

        return global_reference

    def get_warped_image(
        self, reference_image: np.ndarray, moving_image: np.ndarray
    ) -> np.ndarray:

        # Obtain the warp matrix
        warp_matrix = self.find_warp(
            reference_image=reference_image,
            moving_image=self.create_layer(moving_image),
        )

        # Check if the translation is large
        if self.is_translation_large(warp_matrix):
            warnings.warn("Estimated translation is too large")
            warp_matrix = self.warp_matrix

        # Warp the datacube layer
        warped_image = self.warp_feature(img=moving_image, warp_matrix=warp_matrix)

        return warped_image, warp_matrix

    def run(self) -> np.ndarray:
        """
        Method that estimates registrations and warps the datacube
        """

        # Create the global reference
        reference_layer = self.create_layer(self.reference)

        warped_cube = np.zeros_like(self.datacube, dtype=self.datacube.dtype)
        warp_matrices = []
        for index, img in enumerate(self.datacube):
            # Get the warp matrix
            warped_image, warp_matrix = self.get_warped_image(
                reference_image=reference_layer, moving_image=img
            )

            # Save the warp matrix ... copy makes cv2 be less buggy
            warp_matrices.append(warp_matrix.copy())
            warped_cube[index] = warped_image.copy()

        return warped_cube, warp_matrices

    def run_multicore(self) -> np.ndarray:
        """
        Method that estimates registrations and warps the datacube
        using multiple threads
        """
        # Create the global reference
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
                        reference_image=reference_layer,
                        moving_image=img,
                    )
                )

            # Get the results in the same order
            warped_cube = np.zeros_like(self.datacube, dtype=self.datacube.dtype)
            warp_matrices = []
            for index, future in enumerate(futures):
                warped_image, warp_matrix = future.result()
                warped_cube[index] = warped_image
                warp_matrices.append(warp_matrix)

        return warped_cube, warp_matrices
