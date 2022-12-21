import json
import os

actions = dict()

with open('data/kinetics_raw/kinetics_train_label.json') as f:
    jj = json.load(f)
    for _, v in jj.items():
        actions[v["label_index"]] = v["label"]

stats = dict()

directory = "pretrain_eval/kinetics/joint/"
with open(os.path.join(directory, "wrong-samples.txt"), "r") as f:
    lines = f.read().split("\n")
    for l in lines:
        _, pred, true = l.split(",")
        pred, true = int(pred), int(true)
        true_n, pred_n = actions[true], actions[pred]
        if true_n not in stats:
            stats[true_n] = dict()
        if pred_n not in stats[true_n]:
            stats[true_n][pred_n] = 0
        stats[true_n][pred_n] += 1

with open(os.path.join(directory, "stats.txt"), "w") as f:
    f.write("--------------\n")
    for k, v in stats.items():
        v_sorted = {k2: v2 for k2, v2 in sorted(v.items(), key=lambda item: item[1], reverse=True)}
        f.write(f"{k} -> {v_sorted}.\n")
        f.write("--------------\n")

