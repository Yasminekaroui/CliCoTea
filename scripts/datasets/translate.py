import jsonlines
from tqdm import tqdm
from googletrans import Translator
import argparse


def translate_file(ann, dest, keys=["sentence"], max_iteration=1000):
    translator = Translator(
        service_urls=["translate.googleapis.com"]
    )  # , timeout=httpx._config.Timeout(60))

    translated_ann, skipped_ann = [], []
    for i, item in tqdm(enumerate(ann), total=len(ann)):
        iteration = 0
        stop = False
        query = [item[k] for k in keys]
        while not stop:
            try:
                temp = translator.translate(query, src="en", dest=dest)
                stop = True
                for k, trans_sent in enumerate(temp):
                    if isinstance(trans_sent, list):
                        temp_new = []
                        for sen in trans_sent:
                            temp_new.append(sen.text)
                        item[keys[k]] = temp_new
                    elif isinstance(trans_sent, str):
                        item[keys[k]] = trans_sent.text
                translated_ann.append(item)
            except Exception as e:
                print(e)
                iteration += 1
                stop = iteration >= max_iteration
                if stop:
                    skipped_ann.append(i)
                for k, sent in enumerate(query):
                    if isinstance(sent, list):
                        item[keys[k]] = [None] * len(sent)
                    elif isinstance(query, str):
                        item[keys[k]] = None
                translated_ann.append(item)
    return translated_ann, skipped_ann


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--field", nargs="+", type=str, required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    ann = jsonlines.open(args.input_file)
    translated_ann, skipped_ann = translate_file(
        ann, args.lang, key=args.field, max_iteration=args.max_iterations
    )
    with jsonlines.open(args.output_file, "w") as f:
        f.write_all(translated_ann)
    with jsonlines.open(f"skipped_{args.output_file}", "w") as f:
        f.write_all(skipped_ann)
