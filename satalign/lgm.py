import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from satalign.lightglue import (ALIKED, DISK, SIFT, DoGHardNet, LightGlue,
                               SuperPoint)
from satalign.lightglue.utils import rbd
from satalign.main import SatAlign


class LGM(SatAlign):
    def __init__(
        self,
        datacube: np.ndarray,
        reference: np.ndarray,
        feature_model: str = "superpoint",
        matcher_model: str = "lightglue",
        max_num_keypoints: int = 2048,
        device: str = "cpu",
        **kwargs,
    ):
        """

        Args:
            datacube (xr.DataArray): The data cube to be aligned. The data cube
                needs to have the following dimensions: (time, bands, height, width).
            reference (Optional[xr.DataArray], optional): The reference image.
                The reference image needs to have the following dimensions:
                (bands, height, width).
            feature_model (str, optional): The feature extractor model. Defaults to
                'superpoint'. Options are: 'superpoint'. Options are: 'superpoint',
                'disk', 'sift', 'aliked', 'doghardnet'.
            matcher_model (str, optional): The matcher model. Defaults to 'lightglue'.
            max_num_keypoints (int, optional): The maximum number of keypoints. Defaults
                to 2048.
            device (str, optional): The device to use. Defaults to 'cpu'.
        """
        super().__init__(datacube=datacube, reference=reference, **kwargs)

        # General attributes
        self.datacube = datacube
        self.reference = reference

        # Load the feature and matcher models
        self.feature_model, self.matcher_model = self.spatial_setup_model(
            feature_model=feature_model,
            matcher_model=matcher_model,
            max_num_keypoints=max_num_keypoints,
            device=device,
        )
        self.feature_model = self.feature_model.eval().to(device)
        self.matcher_model = self.matcher_model.eval().to(device)
        self.device = device

        # Create the reference points
        self.reference_points = self.get_reference_points()

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

        # Load the reference points
        feats1 = self.reference_points

        # Moving feature image to torch (1 x H x W)
        moving_image_torch = (
            torch.from_numpy(moving_image).float()[None].to(self.device)
        )

        # Get the reference points from the moving image feature
        with torch.no_grad():
            feats0 = self.feature_model.extract(moving_image_torch, resize=None)
            if feats0["keypoints"].shape[1] == 0:
                warnings.warn("No keypoints found in the moving image")
                return self.warp_matrix

        # Run the matcher model
        matches01 = self.matcher_model({"image0": feats0, "image1": feats1})

        # remove batch dimension
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01["matches"]
        points0 = feats0["keypoints"][matches[..., 0]]
        points1 = feats1["keypoints"][matches[..., 1]]

        # if the distance between the points is higher than self.threshold_distance
        # it is considered a bad match
        dist = torch.sqrt(torch.sum((points0 - points1) ** 2, dim=1))
        thres = dist < self.max_translations

        if thres.sum().item() == 0:
            warnings.warn("No matching points found")
            return self.warp_matrix

        p0 = points0[thres]
        p1 = points1[thres]

        # Get the warp matrix
        translation_x = p1[:, 0].mean() - p0[:, 0].mean()
        translation_y = p1[:, 1].mean() - p0[:, 1].mean()
        warp_matrix = np.eye(*self.warp_matrix_size, dtype=np.float32)
        warp_matrix[:2, 2] = [translation_x.item(), translation_y.item()]

        return warp_matrix

    def spatial_setup_model(
        self,
        feature_model: str,
        matcher_model: str,
        max_num_keypoints: int,
        device: str,
    ) -> Tuple[Union[SuperPoint, DISK, SIFT, ALIKED, DoGHardNet], LightGlue]:
        """Setup the model for spatial check

        Args:
            feature_model (str, optional): The feature extractor model. Defaults to
                'superpoint'. Options are: 'superpoint'. Options are: 'superpoint',
                'disk', 'sift', 'aliked', 'doghardnet'.
            matcher_model (str, optional): The matcher model. Defaults to 'lightglue'.
            max_num_keypoints (int, optional): The maximum number of keypoints. Defaults
                to 2048.
            device (str, optional): The device to use. Defaults to 'cpu'.

        Raises:
            ValueError: Unknown feature extractor model
            ValueError: Unknown matcher model

        Returns:
            Tuple[Union[SuperPoint, DISK, SIFT, ALIKED, DoGHardNet], LightGlue]: The
                feature and matcher models
        """

        # Local feature extractor
        if feature_model == "superpoint":
            extractor = (
                SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
            )
        elif feature_model == "disk":
            extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(device)
        elif feature_model == "sift":
            extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(device)
        elif feature_model == "aliked":
            extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(device)
        elif feature_model == "doghardnet":
            extractor = (
                DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(device)
            )
        else:
            raise ValueError(f"Unknown feature extractor {feature_model}")

        # Local feature matcher
        if matcher_model == "lightglue":
            matcher = LightGlue(features=feature_model).eval().to(device)
        else:
            raise ValueError(f"Unknown matcher {matcher_model}")

        return extractor, matcher

    def get_reference_points(self) -> dict:
        """Since the reference image is static, we extract the reference points
        only once. This function extracts the reference points from the reference


        Raises:
            ValueError: No keypoints found in the reference image.

        Returns:
            dict: The reference points from the reference image (metadata)
        """

        # Create the reference layer (H x W) to torch
        reference_layer = self.create_layer(img=self.reference[self.rgb_bands])
        reference_layer_torch = (
            torch.from_numpy(reference_layer).float()[None].to(self.device)
        )

        # Get the reference points from the reference image
        with torch.no_grad():
            feats0 = self.feature_model.extract(reference_layer_torch, resize=None)
            if feats0["keypoints"].shape[1] == 0:
                raise ValueError("No keypoints found in the reference image")

        return feats0
