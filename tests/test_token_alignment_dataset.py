import os
import tempfile
import json
from clicotea.data import TokenAlignmentDataset


def test_token_alignment_dataset(flickr30k_subset_en_de):
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp:
            for obj in flickr30k_subset_en_de:
                tmp.write(json.dumps(obj) + "\n")
            # do stuff with temp file
        dset = TokenAlignmentDataset(
            token_alignment_file=path,
            src_tokenizer_name="bert-base-multilingual-cased",
            tgt_tokenizer_name="bert-base-multilingual-cased",
            align_with_cls=True,
        )
        assert len(dset) == 5
        assert dset[0][2] == [
            (0, 0),  # [CLS] - [CLS]
            (5, 5),  # stand - steht
            (6, 6),  # in - auf
            (8, 8),  # back - Ladefläche
            (1, 1),  # A - Eine
            (2, 2),  # group - Gruppe
            (14, 13),  # cottton - Baumwolle
            (13, 12),  # with - mit
            (4, 4),  # people - Menschen
            (10, 11),  # a - eines
            (14, 19),  # cotton. - Lastwagens.
            (3, 3),  # of - von
            (7, 7),  # the - der
            (11, 19),  # truck - Lastwagens.
            (12, 16),  # filled - beladenen
        ]
    finally:
        os.remove(path)
