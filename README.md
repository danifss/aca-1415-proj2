### Arquitetura de Computadores Avançada (Advanced Computer Architecture)
> Ano lectivo de 2014 / 2015 - Universidade de Aveiro

# Canny Edge Detector using CUDA

In this assignment gray scale images will be processed in order to detect the edges of the image.
The image will be processed for edge detection using the Canny Edge Detector. This algorithm finds the edges in the image in 4 stages: first, a gaussian filter is applied to minimize noise effects; then the gradient of the resulting image is obtained; from the gradient, a “non-maximum suppression” approach determines the best candidates for edges among several neighbors and finally edges are traced using hysteresis.

Develop improved versions of the Canny Detector using the CUDA platform. You may develop (and compare) several versions of your code that use different functionalities of the CUDA device (global memory, shared memory, texture memory, etc.).
