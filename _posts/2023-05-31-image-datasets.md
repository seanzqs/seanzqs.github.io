---
layout: post
title: "AIGC related image dataset compilation"
date: 2023-05-31 10:57:00 +0800
last_modified_at: 2023-05-31 10:57:00 +0800
tags: [aigc,dataset]
---
## image datasets compilation

real image dataset:
* [laion-aesthetics](https://laion.ai/blog/laion-aesthetics/) containing the 600Mil image dataset used by stable diffusion: aesthetics scores >=5  
(download parquet files in data folder and run the following command to start downloading:)  
```
img2dataset --url_list data --input_format "parquet" --url_col "URL" --caption_col "TEXT" --output_format files --output_folder laion_images --processes_count 1 --thread_count 64 --resize_mode no
```
[official img2dataset command examples](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion-aesthetic.md)

stable diffusion generated image dataset：
* [stable-diffusion-wordnet-dataset](https://www.kaggle.com/datasets/astoeckl/stable-diffusion-wordnet-dataset) save file as html, there will be a link to download. 27.8GB
 

midjourney generated image dataset：
* [midjourney-texttoimage](https://www.kaggle.com/datasets/succinctlyai/midjourney-texttoimage) 2022-06, 54MB
* [midjourney-v51-prompts-and-image-links](https://www.kaggle.com/datasets/iraklip/midjourney-v51-prompts-and-image-links) midjourney v5.1, 350MB
* [midjourney-art-week-of-april-9-2023](https://www.kaggle.com/datasets/nikbearbrown/midjourney-art-week-of-april-9-2023) midjourney art images 2023-04-09, 278MB
* [midjourney-v5-prompts-and-links](https://www.kaggle.com/datasets/iraklip/midjourney-v5-prompts-and-links) midjourney v5, 2023-03-15, 571MB

