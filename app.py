"""Streamlit App to show predictions of seasons based on uniform"""
from pathlib import Path

import streamlit as st

from utils import get_image, get_learner

st.set_page_config(page_title="SES Season Detector", page_icon="🍂")
st.image("https://qcloud.dpfile.com/pc/FRdBM9z9EBQaO-sql--xoytCJIs5jxu7hvreDLrwzbuLlNhlT-_tpcDqr48eEAibbKcq9vnEaGy3xLEf-_v_oA.jpg")
st.title("SES Season Detector")
st.write("*by Jeff (Estelle's dad)*")
st.write("---")

# Because model is created with this function, we need to redefine it here
get_season = None

version = st.sidebar.radio("Version", ("v1", "v2"))
learn = get_learner(f"ses_uniforms_season_{version}.pkl")
vocab = learn.dls.vocab

def display_prediction(pic):
    img = get_image(pic)
    with learn.no_bar():
        prediction, idx, probabilities = learn.predict(img)
    col_img, col_pred = st.columns(2)
    col_img.image(img, caption=getattr(pic, "name", None))
    col_pred.write(f"### {prediction}")
    col_pred.metric(f"Probability", f"{probabilities[idx].item()*100:.2f}%")


select = st.radio("How to load pictures?", ["from files", "from URL", "from Samples"])
st.write("---")

if select == "from URL":
    url = st.text_input("url")
    if url:
        display_prediction(url)

elif select == "from Samples":
    for pic in Path(__file__).parent.glob("samples/*.jpg"):
        display_prediction(pic)

else:
    pictures = st.file_uploader("Choose pictures", accept_multiple_files=True)
    for pic in pictures:  # type:ignore # this is an iterable
        display_prediction(pic)
