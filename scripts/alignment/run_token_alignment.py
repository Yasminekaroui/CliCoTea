import argparse
from clicotea.data.prepare_dataset import generate_token_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file", required=True)
    parser.add_argument("--tgt-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--field", default="sentence")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="aneuraz/awesome-align-with-co")
    args = parser.parse_args()

    generate_token_pairs(
        args.src_file,
        args.tgt_file,
        args.output_file,
        args.field,
        args.device,
        args.model_name,
    )
