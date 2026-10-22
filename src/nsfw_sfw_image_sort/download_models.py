import logging
import argparse
import pathlib
import urllib
from urllib.request import urlretrieve
import gdown
from zipfile import ZipFile
import os

nudenet_urls = [
    (
        "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx",
        "320n_nudenet.onnx"
    ),
    (
        "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx",
        "640m_nudenet.onnx"
    ),
]

furry_model_url = "https://drive.google.com/uc?export=download&id=1VIhh0yfhJuXXTFo_0XPnOeQp5ePqnjDW"


def download_default():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="./models/")
    parser.add_argument('--nudenet', action='store_true')
    parser.add_argument('--no-nudenet', dest='nudenet', action='store_false')
    parser.set_defaults(nudenet=True)
    parser.add_argument("--nudenet_model_url", type=str, default=None)
    parser.add_argument('--furry_model', action='store_true')
    parser.add_argument('--no-furry_model', dest='furry_model', action='store_false')
    parser.set_defaults(furry_model=True)
    parser.add_argument('-d', '--debug',help="Print lots of debugging statements",action="store_const", dest="loglevel", const=logging.DEBUG,default=logging.WARNING)
    parser.add_argument('-v', '--verbose',help="Be verbose",action="store_const", dest="loglevel", const=logging.INFO)
    # parse args
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    # make sure dest exists
    dest = pathlib.Path(args.dest)
    dest = dest.parent if dest.is_file() else dest
    dest.mkdir(parents=True, exist_ok=True)
    # download
    if args.nudenet:
        if not args.nudenet_model_url:
            logging.info("NudeNet default, downloading both 320 and 640 res models")
            for nurl,savename in nudenet_urls:
                logging.info(f">>>Downloading data for model: {nurl}")
                urlretrieve(nurl, str((dest / savename).resolve()))
    if args.furry_model:
        logging.info(f"Downloading default furry model at {furry_model_url}")
        save_path = (dest / "convnext_tiny_224_5.zip").resolve()
        gdown.download(furry_model_url, str(save_path), quiet=False)
        # unizp
        with ZipFile(save_path, 'r') as zObject:
            zObject.extractall(str(dest.resolve()))
        # delete zip
        save_path.unlink()

if __name__ == '__main__':
    download_default()