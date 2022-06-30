# Image Search by Japenese-CLIP

This project is for [**OPTiM TECH BLOG**](https://tech-blog.optim.co.jp/).

## Usage

1. Copy images which you want to search (`.jpg` only) to `datasets` dir 
   ```
   mkdir datasets
   cp /path/to/images/*.jpg datasets/
   ```
2. Build image
   ```
   docker build -t clip .
   ```
3. Run container
   ```
   docker run --rm -it -p 8501:8501 clip
   ```
4. Open [localhost:8501](localhost:8501) in browser

## LICENCE

[MIT Licesne](./LICENSE)
