#  Project 5: Fun With Diffusion Models!

# Part A: The Power of Diffusion Models

### Displaying Precomputed Text Embeddings

In this project, I used a DeepFloyd IF diffusion model, a two stage model trained as a text-to-image model, which takes text prompts as input and outputs images that are aligned with the text. To begin with, I instantiated DeepFloyd's stage_1 and stage_2 objects used for generation, as well as several text prompts for sample generation, which were the following: "An oil painting of a snowy mountain village," " A man wearing a hat," and "A rocket ship". I used random seed 1213, and tried different combinations of __ to generate various versions of images for the same three prompts. Here are the results:

stage 1 *num_inference_steps* = 20, stage 2 *num_inference_steps* = 20

| BAIR Image 1 | BAIR Image 2 | BAIR Image 2 | 
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="400" src="paint11.png"> |  <img width="400" src="paint"> | <img width="400" src="bair2.jpg"> |

stage 1 *num_inference_steps* = 40, stage 2 *num_inference_steps* = 50

|<img width="400" src="grove1.jpg"> |  <img width="400" src="grove2.jpg"> | <img width="400" src="bair2.jpg"> |

stage 1 *num_inference_steps* = 5, stage 2 *num_inference_steps* = 5
  
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="400" src=""> |  <img width="400" src="paint3man.png"> | <img width="400" src="paint3rocket.png"> |

stage 1 *num_inference_steps* = 10, stage 2 *num_inference_steps* = 10
  
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="400" src="paint4village.png"> |  <img width="400" src="paint4man.png"> | <img width="400" src="paint4village.png"> |

Thoughts:

I ran the three text prompts, 'an oil painting of a snowy mountain village', 'a man wearing a hat', and 'a rocket ship' with 4 iterations of different num_inference_steps. The first iteration was with num_inference_steps 20 for both Stage 1 and Stage 2, which produced images that were not very realistic and more "cartoon-like", other than the man in the hat, which was somewhat realistic but almost a bit too soft (painting-like). The second iteration was produced with 40 num_inference_steps for Stage 1 and 50 num_inference_steps for Stage 2, which led to a more realistic representation of the three prompts, which were all very well-made, especially the man in the hat -- I found this one to be the most realistic of them all. For my third iteration, I tried 5 num_inference_steps for both Stage 1 and Stage 2, which led to pretty faulty pictures. The oil painting of the snowy mountain village and the rocket ship were both splotchy and had a fair bit of noise, and the man wearing a hat turned out completely faulty, almost as if there were incomplete inference steps made towards the final image. For my last iteration, I used 10 num_inference_steps for Stage 1 and Stage 2, which resulted in 3 good pictures of the prompts. All 3 of the pictures were less cartoon-like than the first two iterations, and were fairly solid representations.

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

| Noisy Image at Timestep 250 | Noisy Image at Timestep 500 | Noisy Image at Timestep 750 |
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

## 1.5: Diffusion Model Sampling

By using the function I made in the previous part, iterative_denoise(image, i_start), and setting i_start to 0 and passing random noise into the image, we can generate completely new images from scratch. Below are 5 results of a "high quality photo", generated using these steps.

<img width="250" src="bair1.jpg">   <img width="250" src="bair2.jpg">  <img width="250" src="bair2.jpg">  <img width="250" src="bair2.jpg">   <img width="250" src="bair2.jpg"> 

## 1.6: Classifier-Free Guidance (CFG)

We were able to create 5 new images from scratch, but we can create even better quality photos using classifier-free guidance, which reduces hallucination by incorporating both an unconditional and conditional noise estimate. By running the UNet model twice—once with a conditional prompt and once with an empty prompt for the unconditional estimate — I blended the estimates using the formula *noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)*. For the unconditional generation, I used "a high quality photo" as the prompt to guide the image synthesis process, ensuring that the resulting images are of high quality compared to those generated without CFG. The images produced using this method showed improvements in clarity and detail compared to just diffusion model sampling from the previous part.
Below are 5 results from using CFG!

<img width="250" src="bair1.jpg">   <img width="250" src="bair2.jpg">  <img width="250" src="bair2.jpg">  <img width="250" src="bair2.jpg">   <img width="250" src="bair2.jpg"> 

## 1.7: Image-to-Image Translation

I applied the Classifier-Free Guidance (CFG) technique from the previous part to edit existing images (rather than creating completely new ones from scratch) by adding noise and then denoising them, leveraging the model's capacity to introduce creative changes. This process, aligned with the SDEdit algorithm, involves noising the original image slightly and then using the iterative_denoise_cfg function to iteratively denoise it, aiming to make subtle edits by forcing the noisy image back onto the natural image manifold. I ran this denoising process at noise levels [1, 3, 5, 7, 10, 20], each reflecting increasing similarity to the original image. The results, labeled by their starting indices, demonstrate a progression of edits, showcasing how the image gradually approximates its original form. Additionally, similar edits were performed on two other test images to validate the robustness of this approach.

## 1.7.1: Editing Hand-Drawn and Web Images

Here are the examples of the 

## 1.7.2: Inpainting



## 1.7.3: Text-Conditional Image-to-image Translation



## 1.8: Visual Anagrams

## 1.9 Hybrid Images


#  Part B: Diffusion Models from Scratch!

# Part 1: Training a Single-Step Denoising UNet

## 1.1: Implementing the UNet



## 1.7 Image-to-image Translation

## 1.7 Image-to-image Translation


