import os
import pathlib
import re
import requests
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Union, Tuple

import streamlit as st
from fastai.learner import load_learner, Learner
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


@contextmanager
def set_posix():
    """To be able to load model in Windows"""
    posix_backup = pathlib.PosixPath
    windows_backup = pathlib.WindowsPath
    try:
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath
        elif os.name == "posix":
            pathlib.WindowsPath = pathlib.PosixPath
        yield
    finally:
        pathlib.PosixPath = posix_backup
        pathlib.WindowsPath = windows_backup


# TODO: Try to cache the learner ... better: cache when using it
# @st.cache(hash_funcs={TensorBase: (lambda learn:hash(learn))})
def get_learner(model_path: PathStr) -> Learner:
    def try_loading(
        missing_implem: List[str] = None, nb_iter=5
    ) -> Tuple[List[str], Learner]:
        if missing_implem is None:
            missing_implem = []
        try:
            with set_posix():
                learner = load_learner(model_path)
        except AttributeError as e:
            m_missing_func = re.match(r"Can't get attribute '(.*?)'", str(e))
            if m_missing_func and nb_iter > 0:
                missing_implementation = m_missing_func.group(1)
                setattr(sys.modules["__main__"], missing_implementation, None)
                missing_implem.append(missing_implementation)
                return try_loading(missing_implem, nb_iter - 1)
            raise
        else:
            return missing_implem, learner

    missing_implem, learner = try_loading()
    if missing_implem:
        print(
            f"Missing function implementation: {missing_implem} => used 'None' instead"
        )

    return learner
