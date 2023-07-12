import json
import os
import tempfile
import torch
from lavis.models import load_model_and_preprocess
from clicotea.data import IRTRDataset


def test_vr_dataset(flickr30k_image_root, coco_image_root, xFlickrCO_subset_de):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, vis_processors, txt_processors = load_model_and_preprocess(
        name="albef_retrieval",
        model_type="flickr",
        is_eval=True,
        device=device,
    )

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp:
            for obj in xFlickrCO_subset_de:
                tmp.write(json.dumps(obj) + "\n")
        dset = IRTRDataset(
            ann_file=path,
            flickr30k_image_root=flickr30k_image_root,
            coco_image_root=coco_image_root,
            vis_processor=vis_processors["eval"],
            txt_processor=txt_processors["eval"],
        )
        inputs = dset[0]
        assert inputs["text_input"] == "der mann trägt eine orange wollmütze"
        inputs = dset[1]
        assert (
            inputs["text_input"] == "motorradfahrer schaut die landschaft "
            "an und überlegt sich den besten weg"
        )
    finally:
        os.remove(path)
