import os
import faiss
from clip import create_dataset, load_models


def create_clip_index(vectors, out_path, nlist=5):
    dim = 512  # vector dimension by CLIP
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.train(vectors)  # clustering
    index.add(vectors)
    faiss.write_index(index, out_path)


def main():
    dataset_dir = 'datasets'
    out_dir = 'output'
    image_list_path = os.path.join(out_dir, 'image_list.txt')
    index_path = os.path.join(out_dir, 'index.faiss')
    os.makedirs(out_dir, exist_ok=True)

    models = load_models()

    dataset = create_dataset(dataset_dir, models)
    image_list = dataset['path_list']
    vectors = dataset['vectors']

    with open(image_list_path, 'w') as f:
        f.writelines(image_list)

    create_clip_index(vectors, index_path)


if __name__ == '__main__':
    main()
