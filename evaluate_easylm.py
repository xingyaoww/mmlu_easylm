import os
import json
import numpy as np
import pandas as pd
from categories import subcategories, categories
import time
import urllib

import requests
from requests.exceptions import Timeout, ConnectionError
import mlxu


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    name=('easylm_model', 'name of the model'),
    ntrain=5,
    data_dir=('data', 'MMLU data directory'),
    save_dir=('results', 'evaluation results directory'),
    lm_server_url=('http://localhost:5007/', 'EasyLM language model server URL'),
    wait_for_ready=(True, 'Wait for EasyLM server to be ready before starting evaluation'),
    prompt_prefix=('', 'prefix string for the prompt'),
    prompt_suffix=('', 'suffix string for the prompt'),
)


def wait_for_ready():
    while True:
        try:
            requests.get(urllib.parse.urljoin(FLAGS.lm_server_url, 'ready'))
            return
        except (Timeout, ConnectionError) as e:
            time.sleep(10)


def get_loglikelihood(inputs):
    prefix, text = zip(*inputs)
    prefix = list(prefix)
    text = list(text)
    response = requests.post(
        urllib.parse.urljoin(FLAGS.lm_server_url, 'loglikelihood'),
        json={'prefix_text': prefix, 'text': text}
    ).json()
    return response['log_likelihood']



choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def eval(subject, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = FLAGS.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        prompt = FLAGS.prompt_prefix + prompt + FLAGS.prompt_suffix

        label = test_df.iloc[i, test_df.shape[1] - 1]

        inputs = [(prompt, 'A'), (prompt, 'B'), (prompt, 'C'), (prompt, 'D')]
        probs = get_loglikelihood(inputs)

        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(argv):
    if FLAGS.wait_for_ready:
        wait_for_ready()

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(FLAGS.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    if not os.path.exists(os.path.join(FLAGS.save_dir, "results_{}".format(FLAGS.name))):
        os.makedirs(os.path.join(FLAGS.save_dir, "results_{}".format(FLAGS.name)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(FLAGS.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: FLAGS.ntrain]
        test_df = pd.read_csv(
            os.path.join(FLAGS.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(subject, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(FLAGS.name)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(FLAGS.name, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                FLAGS.save_dir, "results_{}".format(FLAGS.name), "{}.csv".format(subject)
            ),
            index=None,
        )

    metrics = {
        "subcat_breakdown": {},
        "cat_breakdown": {}
    }
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        metrics["subcat_breakdown"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        metrics["cat_breakdown"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    metrics["weighted_acc"] = weighted_acc
    with open(os.path.join(FLAGS.save_dir, "results_{}".format(FLAGS.name), "metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    mlxu.run(main)
