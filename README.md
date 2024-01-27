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

4. **PremiumImageProcessing Class**
- Unlock advanced image manipulation with the PremiumImageProcessing class. Use the `chroma_key()` method to seamlessly replace pixels in one image with those from another based on a specified color, perfect for creating captivating compositions.
- The `sticker()` method lets you overlay images, allowing for playful additions like emojis or logos to your pictures.

5. **ImageKNNClassifier Class**
- Train the classifier on labeled images and predict the label of a new image based on its nearest neighbors.
- Grouping images based on visual characteristics. Experiment with different `n_neighbors` values to find the optimal balance between precision and efficiency.




## Usage

The RGB Image Processing project empowers users to manipulate digital images in Python. Using classes like RGBImage and ImageProcessingTemplate, users can effortlessly create, modify, and transform images, experimenting with effects like negation, grayscale, and rotation. Additionally, the StandardImageProcessing class introduces a resourceful twist, allowing users to redeem coupons for cost-free image processing operations.

## *License*

*University of California, San Diego (UCSD) - Coursework Project*


