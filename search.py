import faiss
from clip import load_models, text2vectors


def load_image_list(image_list_path):
    with open(image_list_path) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def load_index(index_path):
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    return index


def search(query, index, k=3):
    _, searched_index = index.search(query, k)
    return searched_index


def main():
    models = load_models()
    image_list = load_image_list('output/image_list.txt')
    index = load_index('output/index.faiss')
    texts = ['黒い犬']
    query = text2vectors(texts, models)
    result = search(query, index)
    for img_idx in result[0]:
        print(image_list[img_idx])


if __name__ == '__main__':
    main()
