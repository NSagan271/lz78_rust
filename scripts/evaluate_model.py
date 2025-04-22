from lz_embed.classical import NGramSpectrumEmbedding
from sys import stdout, stderr
from dataclasses import dataclass, field
from transformers.hf_argparser import HfArgumentParser
from lz_embed.transformer_based import TokenizedLZPlusEmbedding
import numpy as np
from relu_embed.embedding import NNEmbedding
import mteb


@dataclass
class Arguments:
    model_type: str = field(metadata=dict(
        choices=["lz", "relu", "ngram"],
        help="What type of embedding model to use"
    ))
    model_save_dir: str = field(metadata=dict(
        help="Directory with the model config and output files"))
    benchmark_list_file: str = field(default="mteb_info/tasks.txt", metadata=dict(
        help="Text file with one MTEB task name per line"
    ))
    output_dir: str = field(default="results/test", metadata=dict(
        help="Directory to store task results"
    ))
    overwrite_results: bool = field(default=True, metadata=dict(
        help="Whether to rerun inference if there are already results corresponding to this model in output_dir"
    ))

def main(args: Arguments):
    print("WARNING: it is recommended to forward stderr to a file when running this script!", file=stderr)

    if args.model_type == "lz":
        model = TokenizedLZPlusEmbedding.from_pretrained(args.model_save_dir)
    elif args.model_type == "relu":
        model = NNEmbedding.from_pretrained(args.model_save_dir)
    elif args.model_type == "ngram":
        model = NGramSpectrumEmbedding.from_pretrained(args.model_save_dir)
    else:
        raise NotImplementedError("Unknown model_type")

    with open(args.benchmark_list_file, "r") as f:
        task_list = [x.strip() for x in f.readlines()]

    outputs = []

    for task in task_list:
        tasks = mteb.get_tasks(tasks=[task])
        evaluation = mteb.MTEB(tasks=tasks)

        results = evaluation.run(
            model, output_folder=args.output_dir + "/" + args.model_save_dir.split("/")[-1],
            show_progress_bar=False,
            overwrite_results=args.overwrite_results
        )

        score = np.mean([results[0].scores[name][0]["main_score"] * 100 for name in results[0].scores])
        outputs.append(str(score))
        print(f"TASK: {task}, SCORE: {score}")
        stdout.flush()
        stderr.flush()

    print(" ".join(outputs))
    stdout.flush()
    stderr.flush()


if __name__ == "__main__":
    parser = HfArgumentParser((Arguments))
    (args, ) = parser.parse_args_into_dataclasses()

    main(args)