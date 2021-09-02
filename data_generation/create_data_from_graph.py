"""
Script for generating synthetic data for FactCC training.

Script expects source documents in `jsonl` format with each source document
embedded in a separate json object.

Json objects are required to contain `id` and `text` keys.
"""

import argparse
import json
import os

from tqdm import tqdm

import augmentation_ops as ops


def load_source_docs(file_path, to_dict=False):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        for i in range(len(data)):
            data[i]["id"] = i

    if to_dict:
        data = {example["id"]: example for example in data}
    return data


def save_data(args, data, name_suffix):
    output_file = os.path.splitext(args.data_file)[0] + "-" + name_suffix + ".jsonl"

    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            example = dict(example)
            example["text"] = example["text"]
            example["claim"] = example["claim"]
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def apply_transformation(data, operation):
    new_data = []
    for example in tqdm(data):
        new_example = operation.transform(example)
        if new_example:
            new_data.append(new_example)
    return new_data


def main(args):
    # load data
    source_docs = load_source_docs(args.data_file, to_dict=False)
    print("Loaded %d source documents." % len(source_docs))

    # create or load positive examples
    print("Creating data examples")
    sclaims_op = ops.SampleFactFacet()
    data = apply_transformation(source_docs, sclaims_op)
    print("Created %s example pairs." % len(data))

    if args.save_intermediate:
        save_data(args, data, "clean")

    data_btrans = []
    data_positive = data + data_btrans


    if not args.augmentations or "objectswap" in args.augmentations:
        print("Creating objectswap examples")
        objectswap_op = ops.FactSwap(swp_label="object")
        data_objectswap = apply_transformation(data_positive, objectswap_op)
        print("Negation %s example pairs." % len(data_objectswap))
        if args.save_intermediate:
            save_data(args, data_objectswap, "objectswap")

    if not args.augmentations or "subjectswap" in args.augmentations:
        print("Creating subjectswap examples")
        subjectswap_op = ops.FactSwap(swp_label="subject")
        data_subjectswap = apply_transformation(data_positive, subjectswap_op)
        print("Negation %s example pairs." % len(data_objectswap))
        if args.save_intermediate:
            save_data(args, data_objectswap, "subjectswap")

    """
    if not args.augmentations or "triggerswap" in args.augmentations:
        print("Creating triggerswap examples")
        triggerswap_op = ops.FactSwap("trigger")
        data_triggerswap = apply_transformation(data_positive, triggerswap_op)
        print("Negation %s example pairs." % len(data_triggerswap))
        if args.save_intermediate:
            save_data(args, data_triggerswap, "objectswap")
    """

    if not args.augmentations or "facetswap" in args.augmentations:
        print("Creating facetswap examples")
        objectswap_op = ops.FacetSwap()
        data_facetswap = apply_transformation(data_positive, objectswap_op)
        print("Negation %s example pairs." % len(data_objectswap))
        if args.save_intermediate:
            save_data(args, data_objectswap, "facetswap")

    # add noise to all
    data_negative = data_objectswap + data_subjectswap + data_facetswap #+ data_triggerswap
    save_data(args, data_negative, "negative")




if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("data_file", type=str, help="Path to file containing source documents.")
    PARSER.add_argument("--augmentations", type=str, nargs="+", default=(),
                        help="List of data augmentation applied to data.")
    PARSER.add_argument("--all_augmentations", action="store_true",
                        help="Flag whether all augmentation should be applied.")
    PARSER.add_argument("--save_intermediate", action="store_true",
                        help="Flag whether intermediate data from each transformation should be saved in separate files.")
    ARGS = PARSER.parse_args()
    main(ARGS)
