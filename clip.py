import glob
import itertools
import os
from PIL import Image
import torch
import japanese_clip as ja_clip


def load_models():
    clip, preprocess = ja_clip.load(
        "rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip")
    tokenizer = ja_clip.load_tokenizer()
    return {
        'clip': clip,
        'preprocess': preprocess,
        'tokenizer': tokenizer,
    }


def text2vectors(texts, models):
    encodings = ja_clip.tokenize(
        texts=texts,
        tokenizer=models['tokenizer'],
    )
    with torch.no_grad():
        vectors = models['clip'].get_text_features(**encodings)
    return vectors.detach().numpy()


def create_dataset(dataset_dir, models, batchsize=50):

    image_path_list = glob.glob(os.path.join(dataset_dir, '*.jpg'))
    vector_list = []
    idx = 0
    while True:
        image_path_batch = list(itertools.islice(image_path_list, idx, idx + batchsize))
        if len(image_path_batch) == 0:
            break
        print('Get vectors from image {} to {}...'.format(idx, idx + batchsize))
        idx += batchsize
        images = [Image.open(image_path) for image_path in image_path_batch]
        processed = torch.cat([models['preprocess'](img).unsqueeze(0) for img in images], dim=0)
        with torch.no_grad():
            vector_list.append(models['clip'].get_image_features(processed))
    image_path_list = [f'{pl}\n' for pl in image_path_list]
    vectors = torch.cat(vector_list, dim=0)
    return {
        'path_list': image_path_list,
        'vectors': vectors.detach().numpy(),
    }
