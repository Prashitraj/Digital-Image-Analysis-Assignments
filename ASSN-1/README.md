Overview

Real-world scenes often have a very large dynamic range (the ratio of brightest to darkest intensities) which can span several orders of magnitude. Such high dynamic range (HDR) images cannot be reproduced directly on conventional displays. For example, if all the intensities of the Memorial Church image are linearly rescaled so the brightest pixels are mapped to the highest displayable intensity, the rest of the image becomes extremely dark; if they are rescaled by larger values, details in darker regions become visible, but those in brighter regions are lost.

For a more naturalistic appearance, the range of intensities has to be compressed to the low dynamic range of the display, while approximately maintaining the appearance of the image. This process is known as tone mapping or dynamic range compression.

Requirements
Linear and logarithmic rescaling

Obtain some HDR test images from Paul Debevec’s HDR page under “Radiance Maps”. Load an HDR image into your program (OpenCV: imread, Matlab: hdrread) and visualize it by linearly rescaling the pixel values. Experiment with different scalings to see the results; include one in which all the intensities fit in the displayable range.

Detail enhancement

You should find that details in the image become weaker, because logarithmic rescaling indiscriminately compresses both large-scale intensity variations and local contrast.

To counteract this effect, implement some of the spatial-domain image enhancement techniques covered in class, such as histogram equalization, unsharp masking, etc. Experiment with applying them to the HDR image to boost image details and obtain a more visually appealing result. Any techniques you apply here should be implemented by yourself instead of using built-in functions.

Comment on whether there is a benefit to working in the log-luminance domain, rather than directly on the original RGB intensities.

Tone mapping algorithms

This will be the main work of the assignment. Implement one of the following tone mapping algorithms:

Reinhard et al., “Photographic Tone Reproduction for Digital Images”. For full credit, compute the Gaussian convolutions using an FFT.

Durand and Dorsey, “Fast Bilateral Filtering for the Display of High-Dynamic-Range Images”. For full credit, use the high-dimensional representation of the bilateral filter introduced in Paris and Durand, “A Fast Approximation of the Bilateral Filter using a Signal Processing Approach”.

Fattal et al., “Gradient Domain High Dynamic Range Compression”. See if you can figure out how to solve the Poisson equation (3) using the Fourier transform. For the gradient attenuation function (Sec. 4), since we have not yet covered Gaussian pyramids, just use a single Gaussian-filtered image.

You should use built-in routines to compute the FFT and IFFT. Compare the results of the implemented algorithm with those of your approach in the previous part, and discuss its advantages and disadvantages.
