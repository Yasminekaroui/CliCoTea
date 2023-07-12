import json
import os
import tempfile
import torch
from lavis.models import load_model_and_preprocess
from clicotea.data import VRDataset


def test_vr_dataset(marvl_image_root, marvl_subset_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, vis_processors, txt_processors = load_model_and_preprocess(
        name="albef_nlvr",
        model_type="nlvr",
        is_eval=True,
        device=device,
    )

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp:
            for obj in marvl_subset_id:
                tmp.write(json.dumps(obj) + "\n")
        dset = VRDataset(
            ann_file=path,
            image_root=marvl_image_root,
            vis_processor=vis_processors["eval"],
            txt_processor=txt_processors["eval"],
        )
        inputs = dset[0]
        assert inputs["label"] == 0  # positive
        assert (
            inputs["text_input"] == "panci di salah satu foto berada di atas kompor "
            "yang tidak menyala, sedangkan di foto lainnya, api di bawah panci menyala"
        )
    finally:
        os.remove(path)
