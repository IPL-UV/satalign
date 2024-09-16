# 

<p align="center">
  <img src="https://huggingface.co/datasets/JulioContrerasH/DataMLSTAC/resolve/main/banner_satalign.png" width="45%">
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

## **Overview** 📊

**Satalign** is a Python package designed for efficient multi-temporal image co-registration. It enables aligning temporal data cubes with reference images using advanced techniques such as Phase Cross-Correlation (PCC), Enhanced Cross-Correlation (ECC), and Local Geometric Matching (LGM). This package facilitates the manipulation and processing of large volumes of Earth observation data efficiently.

## **Key features** ✨
- **Advanced alignment algorithms**: Leverages ECC, PCC, and LGM to accurately align multi-temporal images. 🔍
- **Efficient data cube management**: Processes large data cubes with memory and processing optimizations. 🧩
- **Support for local feature models**: Utilizes models like SuperPoint, SIFT, and more for keypoint matching. 🖥️
- **Parallelization**: Executes alignment processes across multiple cores for faster processing. 🚀

## **Installation** ⚙️
Install the latest version from PyPI:

```bash
pip install satalign
```
To use the `PCC` module, you need to install additional dependencies:

```bash
pip install satalign[pcc]
```
Alternatively, if you already have satalign installed:

```bash
pip install scikit-image
```
To use the `LGM` module, you need to install additional dependencies:

```bash
pip install satalign[deep]
```

## **How to use** 🛠️

### **Align an ee.ImageCollection with `satalign.pcc.PCC`** 🌍

#### **Load libraries**

```python
import ee
import fastcubo
import satalign
import satalign.pcc
import matplotlib.pyplot as plt
from IPython.display import Image, display
```

#### **Auth and Init GEE**

```python
# Initialize depending on the environment
ee.Authenticate()
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com") # project = "name"
```
#### **Dataset**
```python
# Download image collection
table = fastcubo.query_getPixels_imagecollection(
    point=(-75.71260, -14.18835),
    collection="COPERNICUS/S2_HARMONIZED",
    bands=["B2", "B3", "B4", "B8"],
    data_range=["2023-12-01", "2023-12-31"],
    edge_size=256,
    resolution=10,
)
fastcubo.getPixels(table, nworkers=4, output_path="output")
```
#### **Align dataset**
```python
# Create a data cube and select images if desired
s2_datacube = satalign.utils.create_array("output", "datacube.pickle")

# Define reference image
reference_image = s2_datacube.sel(time=s2_datacube.time > "2022-08-03").mean("time")

# Initialize and run PCC model
pcc_model = satalign.pcc.PCC(
    datacube=s2_datacube,
    reference=reference_image,
    channel="mean",
    crop_center=128,
    num_threads=2,
)
# Run the alignment
aligned_cube, warp_matrices = pcc_model.run_multicore()

# Display the warped cube
warp_df = satalign.utils.warp2df(warp_matrices, s2_datacube.time.values)
satalign.utils.plot_s2_scatter(warp_df)
plt.show()
```
<p align="center">
  <img src="https://huggingface.co/datasets/JulioContrerasH/DataMLSTAC/resolve/main/warped_cube.png" width="60%">
</p>


#### **Graphics**

```python
# Display profiles
satalign.utils.plot_profile(
    warped_cube=aligned_cube.values,
    raw_cube=s2_datacube.values,
    x_axis=3,
    rgb_band=[3, 2, 1],
    intensity_factor=1/3000,
)
plt.show()
```
<p align="center">
  <img src="https://huggingface.co/datasets/JulioContrerasH/DataMLSTAC/resolve/main/profile.png" width="100%">
</p>

```python
# Create PNGs and GIF
# Note: The following part requires a Linux environment
# !apt-get install imagemagick
gifspath = satalign.utils.plot_animation1(
    warped_cube=aligned_cube[0:50].values,
    raw_cube=s2_datacube[0:50].values,
    dates=s2_datacube.time[0:50].values,
    rgb_band=[3, 2, 1],
    intensity_factor=1/3000,
    png_output_folder="./output_png",
    gif_delay=20,
    gif_output_file="./animation1.gif",
)
display(Image(filename='animation1.gif'))
```
<p align="center">
  <img src="https://huggingface.co/datasets/JulioContrerasH/DataMLSTAC/resolve/main/s2animation1.gif" width="100%">
</p>

Here's an addition to clarify that `datacube` and `reference_image` have already been defined:

### **Align an Image Collection with `satalign.eec.ECC`** 📚

```python
import satalign.ecc

# Initialize the ECC model
ecc_model = satalign.ecc.ECC(
    datacube=s2_datacube, 
    reference=reference_image,
    gauss_kernel_size=5,
)
# Run the alignment
aligned_cube, warp_matrices = ecc_model.run()
```
### **Align using Local Features with `satalign.lgm.LGM`** 🧮

Here's the updated version with a note about using floating-point values or scaling:

```python
import satalign.lgm

# Initialize the LGM model
lgm_model = satalign.lgm.LGM(
    datacube=datacube / 10_000, 
    reference=reference_image / 10_000, 
    feature_model="superpoint",
    matcher_model="lightglue",
)
# Run the alignment
aligned_cube, warp_matrices = lgm_model.run()
```

In this document, we presented three different examples of how to use SatAlign with PCC, ECC, and LGM for multi-temporal image co-registration. Each example shows how to download an image collection from Google Earth Engine, create a data cube, and align the images using one of the three methods provided by the SatAlign package.