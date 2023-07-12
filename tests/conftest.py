import pytest
import os.path as op


@pytest.fixture
def flickr30k_image_root():
    return op.join(op.dirname(op.abspath(__file__)), "fixtures", "flickr30k-images")


@pytest.fixture
def marvl_image_root():
    return op.join(op.dirname(op.abspath(__file__)), "fixtures", "marvl-images", "id")


@pytest.fixture
def coco_image_root():
    return op.join(op.dirname(op.abspath(__file__)), "fixtures", "coco-images")


@pytest.fixture
def flickr30k_subset():
    return [
        {
            "image": "flickr30k-images/1018148011.jpg",
            "caption": [
                "A group of people stand in the back of a truck filled with cotton.",
                "Men are standing on and about a truck carrying a white substance.",
                "A group of people are standing on a pile of wool in a truck.",
                "A group of men are loading cotton onto a truck",
                "Workers load sheared wool onto a truck.",
            ],
        },
    ]


@pytest.fixture
def flickr30k_subset_de():
    return [
        {
            "image": "flickr30k-images/1018148011.jpg",
            "caption": [
                "Eine Gruppe von Menschen steht auf der Ladefl\u00e4che eines mit Baumwolle beladenen Lastwagens.",
                "M\u00e4nner stehen auf und um einen Lastwagen, der eine wei\u00dfe Substanz transportiert.",
                "Eine Gruppe von Menschen steht auf einem Wollhaufen in einem Lastwagen.",
                "Eine Gruppe von M\u00e4nnern l\u00e4dt Baumwolle auf einen Lastwagen",
                "Arbeiter laden geschorene Wolle auf einen Lastwagen.",
            ],
        },
    ]


@pytest.fixture
def flickr30k_subset_en_de():
    return [
        {
            "src": "A group of people stand in the back of a truck filled with cotton.",
            "tgt": "Eine Gruppe von Menschen steht auf der Ladefl\u00e4che eines mit Baumwolle beladenen Lastwagens.",
            "alignment": [
                [4, 4],
                [5, 5],
                [7, 7],
                [0, 0],
                [1, 1],
                [13, 10],
                [12, 9],
                [3, 3],
                [9, 8],
                [13, 12],
                [2, 2],
                [6, 6],
                [10, 12],
                [11, 11],
            ],
        },
        {
            "src": "Men are standing on and about a truck carrying a white substance.",
            "tgt": "M\u00e4nner stehen auf und um einen Lastwagen, der eine wei\u00dfe Substanz transportiert.",
            "alignment": [
                [2, 1],
                [6, 5],
                [4, 3],
                [0, 0],
                [11, 10],
                [5, 4],
                [11, 11],
                [7, 6],
                [9, 8],
                [10, 9],
                [3, 2],
                [8, 11],
            ],
        },
        {
            "src": "A group of people are standing on a pile of wool in a truck.",
            "tgt": "Eine Gruppe von Menschen steht auf einem Wollhaufen in einem Lastwagen.",
            "alignment": [
                [6, 5],
                [0, 0],
                [8, 7],
                [1, 1],
                [5, 4],
                [13, 10],
                [10, 7],
                [12, 9],
                [3, 3],
                [7, 6],
                [2, 2],
                [11, 8],
            ],
        },
        {
            "src": "A group of men are loading cotton onto a truck",
            "tgt": "Eine Gruppe von M\u00e4nnern l\u00e4dt Baumwolle auf einen Lastwagen",
            "alignment": [
                [6, 5],
                [0, 0],
                [8, 7],
                [1, 1],
                [5, 4],
                [3, 3],
                [7, 6],
                [9, 8],
                [2, 2],
            ],
        },
        {
            "src": "Workers load sheared wool onto a truck.",
            "tgt": "Arbeiter laden geschorene Wolle auf einen Lastwagen.",
            "alignment": [[4, 4], [5, 5], [0, 0], [1, 1], [3, 3], [2, 2], [6, 6]],
        },
    ]


@pytest.fixture
def xvnli_subset_fr():
    return [
        {
            "gold_label": "contradiction",
            "sentence1": "Un homme dans un t-shirt à col bleu qui pose devant quelqu'un tout en tenant un journal hébreu.",
            "sentence2": "Il porte le journal sur sa tête.",
            "captionID": "759822302.jpg#2",
            "pairID": "759822302.jpg#2r1c",
            "Flikr30kID": "759822302",
        }
    ]


@pytest.fixture
def marvl_subset_id():
    return [
        {
            "concept": "39-Panci",
            "language": "id",
            "caption": "Panci di salah satu foto berada di atas kompor yang tidak menyala, sedangkan di foto lainnya, api di bawah panci menyala.",
            "left_img": "39-4.jpg",
            "right_img": "39-8.jpg",
            "annotator_info": {
                "annotator_id": "id_01",
                "country_of_birth": "Indonesia",
                "country_of_residence": "Indonesia",
                "gender": "male",
                "age": 31,
            },
            "chapter": "Basic actions and technology",
            "id": "id-1",
            "label": False,
        }
    ]


@pytest.fixture
def xFlickrCO_subset_de():
    return [
        {
            "sentences": ["Der Mann trägt eine orange Wollmütze."],
            "id": "1007129816",
            "img_path": "1007129816.jpg",
        },
        {
            "sentences": [
                "Motorradfahrer schaut die Landschaft an und überlegt sich den besten Weg."
            ],
            "id": 391895,
            "img_path": "COCO_val2014_000000391895.jpg",
        },
    ]
