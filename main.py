import time
import streamlit as st
from PIL import Image
from clip import load_models, text2vectors
from search import load_image_list, load_index, search


def main():
    st.set_page_config(layout="wide")
    with st.spinner('Loading...'):
        models = load_models()
        image_list = load_image_list('output/image_list.txt')
        index = load_index('output/index.faiss')

    st.title('Image search by Japanese-CLIP')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.form('text_form'):
            search_text = st.text_input('Search Text', '黒い犬')
            button = st.form_submit_button('Search Image')

    if not button or search_text == '':
        st.stop()

    t2v_start = time.time()
    query = text2vectors([search_text], models)
    search_start = time.time()
    searched_index = search(query, index)[0]
    search_end = time.time()
    results = [image_list[idx] for idx in searched_index]
    st.write('Text to Vector: {:.4f}[s]'.format(search_start - t2v_start))
    st.write('Search        : {:.4f}[s]'.format(search_end - search_start))

    cols = [col2, col3, col4]
    for i, img_path in enumerate(results):
        with cols[i]:
            img = Image.open(img_path)
            st.image(img, caption=img_path, use_column_width='always')


if __name__ == '__main__':
    main()
