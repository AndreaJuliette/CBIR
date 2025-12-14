import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os

import streamlit as st
from streamlit_cropper import st_cropper
import torch


from torchvision import transforms
from torchvision.models import efficientnet_b0
import cv2 
from skimage.feature import hog
st.set_page_config(layout="wide")

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("database/cnn.index")
    index.nprobe = 4   
    return index

@st.cache_resource
def load_shape_structure_index():
    index = faiss.read_index("database/shape_structure.index")
    index.nprobe = 1  
    return index

@st.cache_resource
def load_color_shape_structure_index():
    index = faiss.read_index("database/shape_texture_color.index")
    index.nprobe = 8
    return index

@st.cache_resource
def load_model():
    model = efficientnet_b0(pretrained=True)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model

model = load_model()

cnn_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
def extract_cnn_from_pil(img_pil):
    with torch.no_grad():
        img = cnn_preprocess(img_pil).unsqueeze(0)
        emb = model(img)

    emb = emb.numpy().flatten().astype("float32")
    emb = emb / np.linalg.norm(emb)
    return emb

# SHAPE + STRUCTURE (HOG + ORB)

def extract_hog(gray01):
    return hog(
        gray01,
        orientations=14,
        pixels_per_cell=(6, 6),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    ).astype("float32")


def extract_orb(gray01, max_features=500):
    orb = cv2.ORB_create(nfeatures=max_features)
    gray_u8 = (gray01 * 255).astype("uint8")
    _, desc = orb.detectAndCompute(gray_u8, None)
    if desc is None:
        return np.zeros(32, dtype="float32")
    return desc.mean(axis=0).astype("float32")


def l2norm(x, eps=1e-12):
    return x / (np.linalg.norm(x) + eps)


def preprocess_shape_structure_pil(img_pil, size=(128, 128)):

    img_rgb = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = gray.astype("float32") / 255.0
    return gray


def extract_shape_structure_from_pil(img_pil, w_hog=1.0, w_orb=2.0):
    gray01 = preprocess_shape_structure_pil(img_pil)

    hog_feat = l2norm(extract_hog(gray01))
    orb_feat = l2norm(extract_orb(gray01))

    combined = np.concatenate([w_hog * hog_feat, w_orb * orb_feat]).astype("float32")
    combined = l2norm(combined).astype("float32")
    return combined

# COLOR + SHAPE + Texture

def extract_color_shape_texture_from_pil(
    img_pil,
    w_color=1.0,
    w_shape=1.0,
    w_texture=1.0,
    bins=(8, 8, 8),
    P=8,
    R=1
):
    color_feat = extract_color_hist_from_pil(img_pil, bins=bins)
    color_feat = l2norm(color_feat)

    gray = preprocess_shape_structure_pil(img_pil)
    hog_feat = l2norm(extract_hog(gray))

    texture_feat = extract_lbp_from_pil(img_pil, P=P, R=R)
    texture_feat = l2norm(texture_feat)

    combined = np.concatenate([
        w_color * color_feat,
        w_shape * hog_feat,
        w_texture * texture_feat
    ]).astype("float32")

    return l2norm(combined)


# COLOR (HSV histogram)

def extract_color_hist_from_pil(img_pil, bins=(8, 8, 8)):

    rgb = np.array(img_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None, bins,
        [0, 180, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten().astype("float32")
    return hist

def sim_color_corr(h1, h2):

    sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return (sim + 1.0) / 2.0 

# EXTRAER TEXTURA

from skimage.feature import local_binary_pattern

def extract_lbp_from_pil(img_pil, P=8, R=1):
    img_rgb = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128))
    gray = gray.astype("float32") / 255.0

    lbp = local_binary_pattern(gray, P, R, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, P + 3),
        range=(0, P + 2),
        density=True
    )

    return hist.astype("float32")
def sim_texture_l2(h1, h2):
    return 1.0 / (1.0 + np.linalg.norm(h1 - h2))

# RE-RANKING

def rerank_cnn_color(query_img, query_cnn, candidate_indices, image_list,
                     w_cnn=0.9, w_color=0.1, bins=(8, 8, 8)):
    q_color = extract_color_hist_from_pil(query_img, bins=bins)

    scores = []
    for idx in candidate_indices:
        img_path = os.path.join(IMAGES_PATH, image_list[idx])
        cand_img = Image.open(img_path).convert("RGB")

        sim_cnn = float(np.dot(query_cnn, extract_cnn_from_pil(cand_img)))
        sim_cnn = (sim_cnn + 1.0) / 2.0  

        c_color = extract_color_hist_from_pil(cand_img, bins=bins)
        sim_col = sim_color_corr(q_color, c_color)

        score = w_cnn * sim_cnn + w_color * sim_col
        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores]


def rerank_cnn_shape(query_img, candidate_indices, image_list):
    q_shape = extract_shape_structure_from_pil(query_img)
    scores = []

    for idx in candidate_indices:
        img_path = os.path.join(IMAGES_PATH, image_list[idx])
        cand = Image.open(img_path).convert("RGB")
        c_shape = extract_shape_structure_from_pil(cand)

        sim = -np.linalg.norm(q_shape - c_shape)  
        scores.append((idx, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores]

def rerank_cnn_color_texture(
    query_img,
    query_cnn,
    candidate_indices,
    image_list,
    w_cnn=0.7,
    w_color=0.15,
    w_texture=0.15
):
    q_color = extract_color_hist_from_pil(query_img)
    q_texture = extract_lbp_from_pil(query_img)

    scores = []

    for idx in candidate_indices:
        img_path = os.path.join(IMAGES_PATH, image_list[idx])
        cand_img = Image.open(img_path).convert("RGB")

        cnn_feat = extract_cnn_from_pil(cand_img)
        sim_cnn = float(np.dot(query_cnn, cnn_feat))
        sim_cnn = (sim_cnn + 1.0) / 2.0


        c_color = extract_color_hist_from_pil(cand_img)
        sim_color = sim_color_corr(q_color, c_color)

        c_texture = extract_lbp_from_pil(cand_img)
        sim_texture = sim_texture_l2(q_texture, c_texture)

        score = (
            w_cnn * sim_cnn +
            w_color * sim_color +
            w_texture * sim_texture
        )

        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores]

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

# Path in which the images should be located
IMAGES_PATH = os.path.join(FILES_PATH, 'images')
# Path in which the database should be located
DB_PATH = os.path.join(FILES_PATH, 'database')

DB_FILE = 'db.csv' # name of the database

def get_image_list():
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE),encoding="latin1",)
    image_list = list(df.image.values)
    return image_list

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    if (feature_extractor == 'CNN'):

        model_feature_extractor = extract_cnn_from_pil

        indexer = load_faiss_index()
    elif (feature_extractor == 'Forma + Estructura'):
        model_feature_extractor = extract_shape_structure_from_pil
        indexer = load_shape_structure_index()
    elif (feature_extractor == "CNN + Forma + Estructura"):
        model_feature_extractor = extract_cnn_from_pil
        indexer = load_faiss_index()

    elif (feature_extractor == 'CNN + Color'):
        model_feature_extractor = extract_cnn_from_pil
        indexer = load_faiss_index()
    elif (feature_extractor == 'CNN + Color + Textura'):
        model_feature_extractor = extract_cnn_from_pil
        indexer = load_faiss_index()
    
    elif feature_extractor == 'Color + Forma + Textura':
        model_feature_extractor = extract_color_shape_texture_from_pil
        indexer = load_color_shape_structure_index()


    embeddings = model_feature_extractor(img_query)
    vector = np.asarray(embeddings, dtype="float32")[np.newaxis, :]

    print("Vector dim:", vector.shape[1])
    print("Index dim:", indexer.d)
    _, indices = indexer.search(vector, k=n_imgs)

    if (feature_extractor == 'CNN + Color'):
        reranked = rerank_cnn_color(
            query_img=img_query,
            query_cnn=embeddings,
            candidate_indices=indices[0],
            image_list=get_image_list(),
            w_cnn=0.8,
            w_color=0.1
        )
        return reranked[:n_imgs] 
    elif (feature_extractor == 'CNN + Forma + Estructura'):

        reranked = rerank_cnn_shape(img_query, indices[0], get_image_list())
        return reranked[:n_imgs] 
    
    elif feature_extractor == 'CNN + Color + Textura':
        reranked = rerank_cnn_color_texture(
            query_img=img_query,
            query_cnn=embeddings,
            candidate_indices=indices[0],
            image_list=get_image_list(),
            w_cnn=0.7,
            w_color=0.15,
            w_texture=0.15
        )
        return reranked[:n_imgs]

    return indices[0]

def main():
    st.title('CBIR IMAGE SEARCH')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        # TODO: Adapt to the type of feature extraction methods used.
        option = st.selectbox('.', ('CNN', 'Forma + Estructura','CNN + Color','CNN + Forma + Estructura','Color + Forma + Textura','CNN + Color + Textura'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            #X Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option, n_imgs=11)
            image_list = get_image_list()

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[0]]))
                st.image(image, use_column_width = 'always')

            with col4:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[1]]))
                st.image(image, use_column_width = 'always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width = 'always')

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width = 'always')

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width = 'always')

if __name__ == '__main__':
    main()