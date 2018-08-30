# ColorBooked 
Automated generation of coloring book images. Convert an image into a coloring book image via three simple steps:

1. Use Gaussian blurring to smooth out features that would otherwise make the final coloring book image too noisy.
2. K-means clustering to segment the image into N different colors
3. Edge detection on the segmented RGB image

This approach leads to guaranteed closure of all contours in the final outlined image when performing the edge detection step. The final product is the color segmented image and the outlined image for coloring.

Usage:

python /path/to/color_booked.py image.jpg --blur StdDev --colors num_colors

StDev: the standard deviation of the Gaussian blur. Default = min(width px, height px)/500

num_colors: N colors in the final image. Default = 10 colors
