Overview

In class we have discussed many techniques for restoring images that have been degraded by noise and/or blur, from the “classical” averaging filters and Wiener filtering to more recent methods. In this assignment you will experiment with these techniques on some images with synthetic and real degradations.

Requirements
Here are some test images you can use for your results (click for full size):

   

You may use built-in library routines for PSF construction, convolution, median filtering, and the Fourier transform.

Basic denoising

Add Gaussian noise to each test image to create a noticeably noisy image. Then explore the use of mean filters and median filters to reduce the noise: for each type of filter, try different filter sizes, compute the PSNR, and choose the filter size manually (or by some loop) to approximately maximize the PSNR. Then, do the same with salt-and-pepper noise instead of Gaussian noise. Report your observations, including the noisy and maximum-PSNR images and their corresponding PSNRs, and discuss what you can conclude about the effectiveness of the two filters in different cases.

Edge-preserving smoothing

Implement one of the edge-preserving smoothing methods discussed in class:

Anisotropic diffusion (Perona and Malik, 1990)
Total variation denoising (Rudin et al., 1992)
Bilateral filtering (see the survey by Paris et al. 2009). Not permitted if you already implemented it in Assignment 1
Non-local means (Buades et al. 2005)
Demonstrate results on the same noisy images from the previous part.

Deblurring
Blur the clean image with a disk-shaped point spread function 

Then perform deblurring using the Wiener filter. For this, you will need to provide an estimate of the desired image’s spectrum

Real-world image restoration

Here are some real-world images suffering from unknown noise and blur. Your task is to estimate the noise distribution and the point spread function purely from the given image, and then perform deblurring to restore the image as well as possible. This may be quite challenging! Even if you have a great estimate of the PSF, you will probably not get an extremely clean and sharp image since the blur is quite large. Do as well as you can.

You are not required to have a perfectly automated solution. Some manual effort is expected, for example: estimating the noise variance by locating a uniform region or by trial and error; finding an image of a point light source to estimate the PSF; or manually choosing an appropriate PSF model (e.g. a uniform disk) and fitting its parameters to features in the image.

Show results on at least two of these images, or at least one image from these and another blurry image you have taken yourself. Discuss your process in detail in your report.
