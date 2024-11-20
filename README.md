#  Project 5: Fun With Diffusion Models!

# Part A: The Power of Diffusion Models

### Displaying Precomputed Text Embeddings

In this project, I used a DeepFloyd IF diffusion model, a two stage model trained as a text-to-image model, which takes text prompts as input and outputs images that are aligned with the text. To begin with, I instantiated DeepFloyd's stage_1 and stage_2 objects used for generation, as well as several text prompts for sample generation, which were the following: "An oil painting of a snowy mountain village," " A man wearing a hat," and "A rocket ship". I used random seed 1213, and tried different combinations of __ to generate various versions of images for the same three prompts. Here are the results:

* = 0.5, * = 0.5

| BAIR Image 1 | BAIR Image 2 | BAIR Image 2 | 
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="400" src="bair1.jpg"> |  <img width="400" src="bair2.jpg"> | <img width="400" src="bair2.jpg"> |

* = 0.5, * = 0.5

|<img width="400" src="grove1.jpg"> |  <img width="400" src="grove2.jpg"> | <img width="400" src="bair2.jpg"> |

* = 0.5, * = 0.5
  
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="400" src="vlsb1.jpg"> |  <img width="400" src="vlsb2.jpg"> | <img width="400" src="vlsb2.jpg"> |

## Part 1: Sampling Loops 

In this section, I aim to write my own sampling loops, using the pretrained DeepFloyd denoisers, to implement tasks such as producing optical illusions or inpainting images. The sampling loop essentially performs reverse diffusion, aiming to start from pure noise, and use the denoiser to remove noise, and produce a clean image after T timesteps.

## 1.1: Implementing the Forward Process

I first implemented the forward process of diffusion: taking a clean image and adding noise to it, as well as scaling it appropriately. 

I implemented the noisy_im = forward(im, t) function, which adds noise to an image that corresponds to that given timstep t. Shown below are the results of the forward function on an image of the campanile, at noise time steps [250, 500, 750], which progressively add more noise to the image.

| Original Campanile Image | Noise Timestep 250 | Noise Timestep 500 | Noise Timestep 750 |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="250" src="bair1.jpg"> |  <img width="250" src="bair2.jpg"> | <img width="250" src="bair2.jpg"> | <img width="250" src="bair2.jpg"> |

## 1.2: Classical Denoising

We can try to use Gaussian blurring to denoise the noisy images, but as we can see, the results do not perform well at denoising the image and recovering the original photo.

| Noisy Image at Timestep 250 | Noisy Image at  Timestep 500 | Noisy Image at  Timestep 750 |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="250" src="bair1.jpg"> |  <img width="250" src="bair2.jpg"> | <img width="250" src="bair2.jpg"> | 

| Blurred Noise Timestep 250 | Blurred Noise Timestep 500 | Blurred Noise Timestep 750 |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="250" src="bair1.jpg"> |  <img width="250" src="bair2.jpg"> | <img width="250" src="bair2.jpg"> | 

### 1.3: One-Step Denoising

To attempt to denoise the image, I used a pretrained diffusion model called UNet (which was trained on large datasets of clean & noisy pairs of images), which predicts the Gaussian noise added to an image. Using this Gaussian noise, I then subtracted it from the image to obtain an estimate of the original image (after scaling the noise to the size of the image appropriately).

Shown below are the noisy images at timesteps 250, 500, and 750, and the corresponding one-step denoised images using UNet. 

| Noisy Image at Timestep 250 | Noisy Image at Timestep 500 | Noisy Image at Timestep 750 |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="350" src="bair1.jpg"> |  <img width="350" src="bair2.jpg"> | <img width="350" src="bair2.jpg"> | 

| One-Step Denoised Image at Timestep 250 | One-Step Denoised Image at Timestep 500 | One-Step Denoised Image at Timestep 750 |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="350" src="bair1.jpg"> |  <img width="350" src="bair2.jpg"> | <img width="350" src="bair2.jpg"> | 

At higher timesteps, due to more noise, the diffusion model struggles more to accurately estimate the noise added, and thus the estimated original image progressively gets less accurate.

## 1.4: Iterative Denoising

Instead of one-step denoising, iterative denoising is a much more accurate way to accurately estimate and remove the noise to obtain a clean image. Instead of iterating by one timestep, which can become inefficient, we can use strided timesteps and still obtain accurate results due to https://yang-song.net/blog/2021/score/. I created strided_timesteps, 
and then implemented the function iterative_denoise(image, i_start), which 

| Original Campanile Image | Noisy Campanile at Timestep 90 | Noisy Campanile at Timestep 240 |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="350" src="bair1.jpg"> |  <img width="350" src="bair2.jpg"> | <img width="350" src="bair2.jpg"> | 


| Noisy Campanile at Timestep 390 | Noisy Campanile at Timestep 540 | Noisy Campanile at Timestep 690 |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="350" src="bair1.jpg"> |  <img width="350" src="bair2.jpg"> | <img width="350" src="bair2.jpg"> | 


| Iteratively Denoised Campanile | One-Step Denoised Campanile | Gaussian Blurred Campanile |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="350" src="bair1.jpg"> |  <img width="350" src="bair2.jpg"> | <img width="350" src="bair2.jpg"> | 

As we can see, the iteratively denoised and one-step denoised campanile perform much better than the Gaussian-blurred campanile, and the iteratively denoised image more accurately captures details of the campanile than the one-step denoised, albeit not perfectly.

## 1.5 Diffusion Model Sampling

By using the function I made in the previous part, iterative_denoise(image, i_start), and setting i_start to 0 and passing random noise into the image, we can generate completely new images from scratch. Below are 5 results of a "high quality photo", generated using these steps.

<img width="250" src="bair1.jpg">   <img width="250" src="bair2.jpg">  <img width="250" src="bair2.jpg">  <img width="250" src="bair2.jpg">   <img width="250" src="bair2.jpg"> 

## 1.6 Classifier-Free Guidance (CFG)



## 1.7 Image-to-image Translation



## 1.7.1 Editing Hand-Drawn and Web Images

## 1.7.2 Inpainting

## 1.7.3 Text-Conditional Image-to-image Translation

## 1.8 Visual Anagrams

## 1.9 Hybrid Images


#  Part 2. Recover Homographies

After shooting these pairs of images, I selected 8 correspondence points for each image.
I then wrote a function to compute the homography matrix, given a set of 4+ points for two images. 
Here's an example of correspondence points I picked for the nature images:
