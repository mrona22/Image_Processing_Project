# Image_Processing_Project

## Overview

This Python project includes classes for working with RGB images and implementing basic image processing operations.

## Parts

1. **RGBImage Class:**
- The `RGBImage` class represents an RGB image and provides methods for working with pixel data.
- The class is initialized with a 3D matrix of pixels, where each pixel is represented by a list of three integers (RGB values).
- Various methods such as `size()`, `get_pixels()`, `copy()`, `get_pixel()`, and `set_pixel()` are implemented for manipulating the image data.

 
3. **ImageProcessingTemplate Class:**
- The `ImageProcessingTemplate` class serves as a template for image processing operations.
- It includes methods like `get_cost()`, `negate()`, `grayscale()`, and `rotate_180()`.
- The `negate()`, `grayscale()`, and `rotate_180()` methods return new instances of the `RGBImage` class representing the processed images.
  
  
3. **StandardImageProcessing Class:**
- The `StandardImageProcessing` class extends `ImageProcessingTemplate` and introduces additional functionality.
- It includes a `redeem_coupon()` method, allowing the user to redeem a coupon for a specified number of free image processing operations.
- The `negate()`, `grayscale()`, and `rotate_180()` methods in this class update the cost attribute based on the number of times the operations are performed.


## Usage

The RGB Image Processing project empowers users to manipulate digital images in Python. Using classes like RGBImage and ImageProcessingTemplate, users can effortlessly create, modify, and transform images, experimenting with effects like negation, grayscale, and rotation. Additionally, the StandardImageProcessing class introduces a resourceful twist, allowing users to redeem coupons for cost-free image processing operations.

## *License*

*University of California, San Diego (UCSD) - Coursework Project*


