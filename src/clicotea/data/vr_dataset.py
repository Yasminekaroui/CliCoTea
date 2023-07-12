import os
import jsonlines
from PIL import Image
from torch.utils.data import Dataset


def _create_entry(item):
    entry = {
        "question_id": item["question_id"],
        "image_0": item["image_0"],
        "image_1": item["image_1"],
        "caption": item["caption"],
        "concept": item["concept"],
        "label": item["label"],
    }
    return entry


def _load_dataset(annotations_path):
    """Load entries"""
    with jsonlines.open(annotations_path) as reader:
        # Build an index which maps image id with a list of hypothesis annotations.
        items = []
        count = 0
        for annotation in reader:
            dictionary = {}
            dictionary["image_0"] = annotation["left_img"]
            dictionary["image_1"] = annotation["right_img"]
            dictionary["question_id"] = count
            dictionary["caption"] = str(annotation["caption"])
            dictionary["label"] = int(annotation["label"])
            dictionary["concept"] = str(annotation["concept"])
            # dictionary["scores"] = [1.0]
            items.append(dictionary)
            count += 1

    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries


class VRDataset(Dataset):
    def __init__(self, ann_file, image_root, vis_processor, txt_processor):
        """
        ann_file (string): path the annotation file
        image_root (string): Root directory of images
        vis_processor (callable, optional): A function/transform that takes in an PIL
            image
        txt_processor (callable, optional): A function/transform that takes in a string
        """
        self.entries = _load_dataset(ann_file)
        self.vis_processor = vis_processor
        self.txt_processor = txt_processor
        self.image_root = os.path.join(image_root)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        images_path = os.path.join(self.image_root, "images", entry["concept"])
        image0_path = os.path.join(images_path, entry["image_0"])
        image0 = self.vis_processor(Image.open(image0_path).convert("RGB"))
        image1_path = os.path.join(images_path, entry["image_1"])
        image1 = self.vis_processor(Image.open(image1_path).convert("RGB"))
        sentence = self.txt_processor(entry["caption"])
        label = entry["label"]
        return {
            "image0": image0,
            "image1": image1,
            "text_input": sentence,
            "label": label,
        }
