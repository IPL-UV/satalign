# 

<p align="center">
  <img src="https://huggingface.co/datasets/JulioContrerasH/DataMLSTAC/resolve/main/banner_satalign.png" width="100%">
</p>

<p align="center">
    <em>A Python package for efficient multi-temporal image co-registration</em> 🚀
</p>

<p align="center">
<a href='https://pypi.python.org/pypi/satalign'>
    <img src='https://img.shields.io/pypi/v/satalign.svg' alt='PyPI' />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://pycqa.github.io/isort/" target="_blank">
    <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="isort">
</a>
</p>

---

**GitHub**: [https://github.com/IPL-UV/satalign](https://github.com/IPL-UV/satalign) 🌐

**PyPI**: [https://pypi.org/project/satalign/](https://pypi.org/project/satalign/) 🛠️

---

## Overview 📊

**Satalign** is a Python package designed for efficient multi-temporal image co-registration. It enables aligning temporal data cubes with reference images using advanced techniques such as Phase Cross-Correlation (PCC), Enhanced Cross-Correlation (ECC), and Local Features Matching (LGM). This package facilitates the manipulation and processing of large volumes of Earth observation data efficiently.

## Key features ✨
- **Advanced alignment algorithms**: Leverages ECC, PCC, and LGM to accurately align multi-temporal images. 🔍
- **Efficient data cube management**: Processes large data cubes with memory and processing optimizations. 🧩
- **Support for local feature models**: Utilizes models like SuperPoint, SIFT, and more for keypoint matching. 🖥️
- **Parallelization**: Executes alignment processes across multiple cores for faster processing. 🚀

## Installation ⚙️
Install the latest version from PyPI:


```bash
pip install satalign
```

## How to use 🛠️


### Align an ee.ImageCollection with `satalign.PCC` 🌍

```python
import fastcubo
import ee
import satalign

ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

# Download an image collection
table = fastcubo.query_getPixels_imagecollection(
    point=(-75.71260, -14.18835),
    collection="COPERNICUS/S2_HARMONIZED",
    bands=["B2", "B3", "B4", "B8"],
    data_range=["2018-01-01", "2024-12-31"],
    edge_size=256, 
    resolution=10, 
)

fastcubo.getPixels(table, nworkers=10, output_path="output/aligned_images/s2")

# Create the data cube
s2_datacube = satalign.utils.create_array("output/aligned_images/s2", "output/datacube_pcc.pickle")

# Define the reference image
reference_image = s2_datacube.sel(time=s2_datacube.time > "2024-01-03").mean("time")

# Initialize the PCC model
pcc_model = satalign.PCC(
    datacube=s2_datacube, # T x C x H x W
    reference=reference_image, # C x H x W
    channel="mean",
    crop_center=128,
    num_threads=2,
)

# Run the alignment on multiple cores
aligned_cube, warp_matrices = pcc_model.run_multicore()

```

### Align an Image Collection with `satalign.ECC` 📚

```python
import fastcubo
import ee
import satalign

ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

# Download an image collection
table = fastcubo.query_getPixels_imagecollection(
    point=(51.079225, 10.452173),
    collection="COPERNICUS/S2_HARMONIZED",
    bands=["B4", "B3", "B2"],
    data_range=["2016-06-01", "2017-07-01"],
    edge_size=128,
    resolution=10,
)

fastcubo.getPixels(table, nworkers=4, output_path="output/aligned_images/ecc")

# Create the data cube
s2_datacube = satalign.utils.create_array("output/aligned_images/ecc", "output/datacube_ecc.pickle")

# Define the reference image
reference_image = s2_datacube.isel(time=0)

# Initialize the ECC model
ecc_model = satalign.ECC(
    datacube=s2_datacube, 
    reference=reference_image,
    gauss_kernel_size=3,
)

# Run the alignment
aligned_cube, warp_matrices = ecc_model.run()
```
### Align using Local Features with `satalign.LGM` 🧮

```python
import fastcubo
import ee
import satalign

ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

# Download an image collection
table = fastcubo.query_getPixels_imagecollection(
    point=(-76.5, -9.5),
    collection="NASA/NASADEM_HGT/001",
    bands=["elevation"],
    edge_size=128,
    resolution=90
)

fastcubo.getPixels(table, nworkers=4, output_path="output/aligned_images/lgm")

# Create the data cube
datacube = satalign.utils.create_array("output/aligned_images/lgm", "output/datacube_lgm.pickle")

# Define the reference image
reference_image = datacube.isel(time=0)

# Initialize the LGM model
lgm_model = satalign.LGM(
    datacube=datacube, 
    reference=reference_image, 
    feature_model="superpoint",
    matcher_model="lightglue",
)

# Run the alignment
aligned_cube, warp_matrices = lgm_model.run()
```

In this document, we presented three different examples of how to use SatAlign with PCC, ECC, and LGM for multi-temporal image co-registration. Each example shows how to download an image collection from Google Earth Engine, create a data cube, and align the images using one of the three methods provided by the SatAlign package.