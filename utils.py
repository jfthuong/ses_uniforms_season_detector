import requests
import tempfile
from pathlib import Path
from typing import Union

import streamlit as st
from fastai.vision.core import PILImage


PathStr = Union[str, Path]


@st.cache
def get_image(img: PathStr) -> PILImage:
    """Get picture from either a path or URL"""
    if str(img).startswith("http"):
        with tempfile.TemporaryDirectory() as tmpdirname:
            dest = Path(tmpdirname) / str(img).split("?")[0].rpartition("/")[-1]

            # NOTE: to be replaced by download(url, dest=dest) [from unpackai.utils]
            with requests.get(str(img)) as resp:
                resp.raise_for_status()
                dest.write_bytes(resp.content)

            return PILImage.create(dest)
    else:
        return PILImage.create(img)
