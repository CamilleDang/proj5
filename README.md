#  Project 5: Fun With Diffusion Models!

# Part A: The Power of Diffusion Models

## Displaying Precomputed Text Embeddings

Using blah blah

| BAIR Image 1 | BAIR Image 2 | 
|:-------------------------:|:-------------------------:|
|<img width="400" src="bair1.jpg"> |  <img width="400" src="bair2.jpg"> |

| Grove Image 1 | Grove Image 2 | 
|:-------------------------:|:-------------------------:|
|<img width="400" src="grove1.jpg"> |  <img width="400" src="grove2.jpg"> |

| VLSB Hallway Image 1 | VLSB Hallway Image 2 | 
|:-------------------------:|:-------------------------:|
|<img width="400" src="vlsb1.jpg"> |  <img width="400" src="vlsb2.jpg"> |




#  Part 2. Recover Homographies

After shooting these pairs of images, I selected 8 correspondence points for each image.
I then wrote a function to compute the homography matrix, given a set of 4+ points for two images. 
Here's an example of correspondence points I picked for the nature images:
