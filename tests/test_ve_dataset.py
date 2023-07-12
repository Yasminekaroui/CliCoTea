import json
import os
import tempfile
import torch
from lavis.models import load_model_and_preprocess
from clicotea.data import VEDataset


def test_ve_dataset(flickr30k_image_root, xvnli_subset_fr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, vis_processors, txt_processors = load_model_and_preprocess(
        name="albef_classification",
        model_type="ve",
        is_eval=True,
        device=device,
    )

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp:
            for obj in xvnli_subset_fr:
                tmp.write(json.dumps(obj) + "\n")
        dset = VEDataset(
            ann_file=path,
            image_root=flickr30k_image_root,
            vis_processor=vis_processors["eval"],
            txt_processor=txt_processors["eval"],
        )
        inputs = dset[0]
        assert inputs["label"] == 0  # contradiction
        assert inputs["text_input"] == "il porte le journal sur sa tête"
    finally:
        os.remove(path)
