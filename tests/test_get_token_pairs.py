import torch
from transformers import AutoModel, AutoTokenizer
from clicotea.data.prepare_dataset import get_word_pairs, load_sentences


def test_get_token_pairs(flickr30k_subset, flickr30k_subset_de):
    # load sentences
    sentences_src = load_sentences(flickr30k_subset, item="caption")
    sentences_tgt = load_sentences(flickr30k_subset_de, item="caption")

    # load models
    # for awesome-align, use "aneuraz/awesome-align-with-co"
    # for mBERT, use "bert-base-multilingual-cased"
    model_name = "aneuraz/awesome-align-with-co"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # test the following sentence pair
    # en: A group of people stand in the back of a truck filled with cotton.
    # de: Eine Gruppe von Menschen steht auf der Ladefläche eines mit Baumwolle
    #   beladenen Lastwagens.
    i = 0
    s_src, s_tgt = sentences_src[i], sentences_tgt[i]
    word_pairs = get_word_pairs(s_src, s_tgt, model, tokenizer, device)
    assert list(word_pairs) == [
        (4, 4),
        (5, 5),
        (7, 7),
        (0, 0),
        (1, 1),
        (13, 10),
        (12, 9),
        (3, 3),
        (9, 8),
        (13, 12),
        (2, 2),
        (6, 6),
        (10, 12),
        (11, 11),
    ]
