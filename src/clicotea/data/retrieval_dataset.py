import jsonlines
import os
from PIL import Image
from torch.utils.data import Dataset


class RetrievalDataset(Dataset):
    def __init__(
        self,
        ann_file,
        flickr30k_image_root,
        coco_image_root,
        vis_processor,
        txt_processor,
    ):
        """
        ann_file (string): path the annotation file
        flickr30k_image_root (string): Root directory of images from Flickr30k
        coco_image_root (string): Root directory of images from Coco
         vis_processor (callable, optional): A function/transform that takes in an PIL
            image
        txt_processor (callable, optional): A function/transform that takes in a string
        """
        self.ann = [obj for obj in jsonlines.open(ann_file)]
        self.text = [txt_processor(obj["sentences"][0]) for obj in self.ann]
        self.image = []
        for obj in self.ann:
            image_filename = obj["img_path"]
            if "COCO" in image_filename:
                image_path = os.path.join(coco_image_root, image_filename)
            else:
                image_path = os.path.join(flickr30k_image_root, image_filename)
            self.image.append(vis_processor(Image.open(image_path).convert("RGB")))
        # self.image_root = {
        #     "flickr30k": flickr30k_image_root,
        #     "coco": coco_image_root,
        # }
        # self.vis_processor = vis_processor
        # self.txt_processor = txt_processor

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # ann = self.ann[index]
        # image_filename = ann["img_path"]
        # if "COCO" in image_filename:
        #     image_path = os.path.join(self.image_root["coco"], image_filename)
        # else:
        #     image_path = os.path.join(self.image_root["flickr30k"], image_filename)
        # image = self.vis_processor(Image.open(image_path).convert("RGB"))
        # sentence = self.txt_processor(ann["sentences"][0])  # caption

        return {
            "image": self.image[index],
            "text_input": self.text[index],
        }
