from argparse import ArgumentParser
from collections import Counter,defaultdict
from copy import copy
import os
from tqdm import tqdm

import pandas as pd
import numpy as np
from pymer4.models import Lmer
import torch

from languagemodels import LMFactory, TokenizerFactory
from constants import config2size
from utils import get_device, score_text

device = get_device()

print(f"Using device: {device}" + f" ({torch.cuda.get_device_name()})" if device == "cuda" else "")

pd.options.mode.chained_assignment = None  # default='warn'


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--babylm_train_file", type=str, help="Path to the babylm data (for frequency dict)")
    parser.add_argument("--models_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--naturalstories_dir", type=str)
    parser.add_argument("--checkpoint", type=str)
    
    args = parser.parse_args()

    assert os.path.exists(args.babylm_train_file), f"No train data found at {args.babylm_train_file}"
    assert os.path.exists(args.models_dir), f"Models path {args.models_dir} does not exist!"
    assert os.path.exists(args.naturalstories_dir), f"Naturalstories corpus not found at {args.naturalstories_RTS_dir}"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def main():

    regr_columns = [
        # 'log_RT',
        'nItem',
        "sent_pos",
        # "freq",
        'length', 
        # 'log_freq'
        'sent_id', 'sent_pos'
    ]

    args = parse_args()

    word_freqs = Counter()

    with open(args.babylm_train_file, "r") as f:
        [word_freqs.update(line.split()) for line in f.read().split("\n")]

    natural_stories_path = os.path.join(args.naturalstories_dir, "all_stories.tok")

    df = pd.read_csv(natural_stories_path, sep='\t', header=0)

    def get_story_text(story_id: int):
        assert story_id > 0 and story_id <= 10
        story = list(df.loc[df["item"] == story_id].word)
        return story

    stories = [get_story_text(i) for i in range(1,11)]

    OUTPUT_DIR = os.path.join(args.output_dir, args.checkpoint)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # compile lmer predictors
    for config in tqdm(os.listdir(args.models_dir), desc="Scoring sequences..."):
        print(config)
        dfs = []
        for seed in ["0", "13"]:
            run_dir = os.listdir(os.path.join(args.models_dir, config, seed))[0]
            checkpoint_path = os.path.join(args.models_dir, config, seed, run_dir, args.checkpoint)
            config_path = os.path.join(checkpoint_path, "config.json")
            tokenizer = TokenizerFactory.get_tokenizer("pretrained-tokenizer-fast", tokenizer_name_or_path=checkpoint_path)
            model, _ = LMFactory.get_lm("opt-with-alibi", pre_trained=True, config_name_or_path=config_path, model_name_or_path=checkpoint_path)
            model.to(device)
            model.eval()
            
            for i in range(len(stories)):
                df = score_text(model, tokenizer, copy(stories[i]))
                df["seed"] = [seed for _ in range(len(df))]

                # story id for merging
                df["item"] = [i+1 for _ in range(len(df))]
                
                # add sentence position
                sent_pos, sent_id = [], []
                curr_sent_pos, curr_sent_id = 1, 1
                for word in df["word"]:
                    sent_pos.append(curr_sent_pos)
                    sent_id.append(curr_sent_id)
                    curr_sent_pos += 1
                    if word.endswith("."):
                        curr_sent_pos = 1
                        curr_sent_id += 1
                df["sent_id"] = sent_id
                df["sent_pos"] = sent_pos

                # zone needed for merging with reading_time data
                df["zone"] = range(1, len(df)+1)

                # add word length & frequency
                df["length"] = [len(word) for word in df["word"]]
                df["freq"] = [word_freqs[word] for word in df["word"]]
                
                dfs.append(df)

        df_config = pd.concat(dfs)
        
        # average over seeds
        len_pre = len(df_config)
        df_0 = df_config.loc[df_config["seed"]=="0"]
        df_13 = df_config.loc[df_config["seed"]=="13"]
        df_0["surprisal"] = (df_0["surprisal"].to_numpy() + df_13["surprisal"].to_numpy())/2
        df_config = df_0.drop("seed", axis=1)

        assert len(df_config) == len_pre/2, f"{len(df_config)} != {len_pre/2}"
        
        # save to csv
        df_config.to_csv(os.path.join(OUTPUT_DIR, f"{config}.tsv"), sep="\t")

    # Load per-subject RTs
    subject_rts_path = os.path.join(args.naturalstories_dir, "processed_RTs.tsv")
    subject_rts = pd.read_csv(subject_rts_path, sep="\t")

    init_len = len(subject_rts)

    for i, data_file in enumerate(os.listdir(OUTPUT_DIR)):
        if not data_file.startswith("layers_"):
            continue
        df = pd.read_csv(os.path.join(OUTPUT_DIR, data_file), sep="\t", index_col=0)
        df = df.drop(["word"], axis=1)
        config = data_file[:-4]
        config_short = f"{config.split('_')[1]}_{config.split('_')[3]}"
        df = df.rename(columns={"surprisal": f"srp_{config_short}"})
        if i != 0:
            df = df.drop(["sent_id", "sent_pos", "length", "freq"], axis=1)
        subject_rts = pd.merge(subject_rts, df, on=["zone", "item"], how="left")
        assert len(subject_rts) == init_len

    subject_rts.to_csv(os.path.join(OUTPUT_DIR, "subject_rts_with_surprisal.tsv"), sep="\t")

    print("Normalizing data...")

    subject_rts = subject_rts[subject_rts["freq"]>0] # drop items that are not in the vocabulary

    regr_columns = regr_columns + [column for column in subject_rts.columns if column.startswith("srp_")]

    subject_rts["log_RT"] = np.log(subject_rts["RT"])
    subject_rts["log_freq"] = np.log(subject_rts["freq"])
    for column in regr_columns:
        subject_rts[f"orig_{column}"] = copy(subject_rts[column])
        subject_rts[column] = subject_rts[column] - np.mean(subject_rts[column]) # maybe divide by std

    print("Done.")

    # splits = {}
    # splits["all"] = df

    # split data in two sets
    # could also do a 70%/30% but in terms of subjects/items

    # worker_ids = df["WorkerId"].unique().tolist()
    # items = df["item"].unique().tolist()
    # split_idx_item = int(len(items)/2)

    # workers_a = np.random.choice(worker_ids, int(len(worker_ids)/2))
    # workers_b = [worker_id for worker_id in worker_ids if not worker_id in workers_a]

    # items_a, items_b = [1,2,3,4,5], [6,7,8,9,10]

    # splits["WorkerA_ItemA"] = df[df["WorkerId"].isin(workers_a) & df["item"].isin(items_a)]
    # splits["WorkerA_ItemB"] = df[df["WorkerId"].isin(workers_a) & df["item"].isin(items_b)]
    # splits["WorkerB_ItemA"] = df[df["WorkerId"].isin(workers_b) & df["item"].isin(items_a)]
    # splits["WorkerB_ItemB"] = df[df["WorkerId"].isin(workers_b) & df["item"].isin(items_b)]
    # splits["AA_BB"] = pd.concat([splits["WorkerA_ItemA"],splits["WorkerB_ItemB"]])
    # splits["AB_BA"] = pd.concat([splits["WorkerA_ItemB"],splits["WorkerB_ItemA"]])

    # for split, df in splits.items():
    #     print(split, len(df))

    configs = [col for col in subject_rts.columns if col.startswith("srp_")]
    # print(configs)

    # base model without surprisal
    result = defaultdict(list)

    print("Fitting base model...")

    base_model = Lmer("log_RT ~ length + log_freq + sent_pos + sent_id + nItem + (1|word) + (1|WorkerId) + (1|item)", data=subject_rts)
    base_model.fit()
    base_Loglike = base_model.logLike
    # base_logLike_avg = base_Loglike/len(subject_rts)

    # result["config"].append("base")
    # result["logLike"].append(base_Loglike)
    # result["logLike_avg"].append(base_logLike_avg)
    # result["delta_logLike"].append(0.0)
    # result["delta_logLike_avg"].append(0.0)
    # result["size"].append(0)

    print("Done.")

    # models with surprisal
    for config in tqdm(configs, desc="Fitting lmer models..."):
        print(config)
        config_short = config[4:].replace("_", "*", 1)
        model = Lmer(f"log_RT ~ {config} + length + log_freq + sent_pos + sent_id + nItem + (1|word) + (1|WorkerId) + (1|item)", data=subject_rts)
        model.fit()
        logLike = model.logLike
        loglike_avg = logLike / len(subject_rts)
        delta_logLike = np.abs(base_Loglike - logLike)
        delta_logLike_avg = delta_logLike / len(subject_rts)
        result["config"].append(config)
        # result["surprisal_avg"].append(subject_rts[f"orig_{config}"].mean())
        result["logLike"].append(logLike)
        result["logLike_avg"].append(loglike_avg)
        result["delta_logLike"].append(delta_logLike)
        result["delta_logLike_avg"].append(delta_logLike_avg)
        result["size"].append(config2size[config_short])

    print("Done.")

    df_lmer = pd.DataFrame.from_dict(result)
    df_lmer.to_csv(os.path.join(OUTPUT_DIR, "lmer_results_all.tsv"), sep="\t", index=0)


if __name__ == "__main__":
    main()