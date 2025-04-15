from lz_embed.classical import BasicNGramSpectrum, AlphabetInfo, NGramSpectrumEmbedding
from lz_embed.transformer_based import TokenizedLZPlusEmbedding, WeightType
import matplotlib.pyplot as plt
import mteb
from sys import stdout, stderr
import numpy as np

def main():
    print("WARNING: it is recommended to forward stderr to a file when running this script!", file=stderr)

    # CHANGE THIS ##############
    model = NGramSpectrumEmbedding(
        alpha_info=AlphabetInfo(valid_character_string="abcdefghijklmnopqrstuvwxyz"),
        n=2, lowercase=True, normalize=True, normalized_subspace=True
    )
    # model.register_subspace(file="data/ngram-pca-tmp/subspace_1024.pkl")

    # model = TokenizedLZPlusEmbedding( 
    #     inner_model_name="BAAI/bge-base-en-v1.5",
    #     output_dir="data/object2",
    #     compute_device="cuda:7",
    #     weight_type=WeightType.LOG_LOSS,
    #     pca_dim=256,
    #     pca=True,
    #     overwrite_objects=False
    # )
    # model.spa.set_inference_config(
    #     lb=1e-3,
    #     gamma=1/model.charmap.alphabet_size(),
    #     ensemble_type="entropy",
    #     ensemble_n=6,
    #     backshift_ctx_len=10
    # )
    ############################

    with open("python_experiments/tasks.txt", "r") as f:
        task_list = [x.strip() for x in f.readlines()]

    outputs = []

    for task in task_list:
        tasks = mteb.get_tasks(tasks=[task])
        evaluation = mteb.MTEB(tasks=tasks)

        results = evaluation.run(
            model, output_folder=f"results/test",
            show_progress_bar=False,
            overwrite_results=True
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
    main()