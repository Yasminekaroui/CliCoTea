import jsonlines
import os
from PIL import Image
from torch.utils.data import Dataset


class VEDataset(Dataset):
    def __init__(self, ann_file, image_root, vis_processor, txt_processor):
        """
        ann_file (string): path the annotation file
        image_root (string): Root directory of images
        vis_processor (callable, optional): A function/transform that takes in an PIL
            image
        txt_processor (callable, optional): A function/transform that takes in a string
        """
        self.ann = [obj for obj in jsonlines.open(ann_file)]
        self.vis_processor = vis_processor
        self.txt_processor = txt_processor
        self.image_root = image_root
        self.labels = {"entailment": 2, "neutral": 1, "contradiction": 0}

        self._filter_ann()

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        if "Flickr30K_ID" in ann:
            image_path = os.path.join(self.image_root, f"{ann['Flickr30K_ID']}.jpg")
        elif "Flikr30kID" in ann:
            image_path = os.path.join(self.image_root, f"{ann['Flikr30kID']}.jpg")
        else:
            raise NotImplementedError

        image = self.vis_processor(Image.open(image_path).convert("RGB"))
        sentence = self.txt_processor(ann["sentence2"])  # hypothesis
        return {
            "image": image,
            "text_input": sentence,
            "label": self.labels[ann["gold_label"]],
        }

    def _filter_ann(self):
        to_delete = []
        for i, ann in enumerate(self.ann):
            if ann["gold_label"] not in self.labels.keys():
                to_delete.append(i)

        for i in reversed(to_delete):
            del self.ann[i]
