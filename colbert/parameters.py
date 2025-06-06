import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVED_CHECKPOINTS = [32*1000, 100*1000, 150*1000, 200*1000, 250*1000, 300*1000, 400*1000]
SAVED_CHECKPOINTS += [10*1000, 20*1000, 30*1000, 40*1000, 50*1000, 60*1000, 70*1000, 80*1000, 90*1000]
SAVED_CHECKPOINTS += [25*1000, 50*1000, 75*1000]

SAVED_CHECKPOINTS += [0, 200, 400, 800, 1200, 1500, 3000, 4000, 5000, 7000, 9000]

SAVED_CHECKPOINTS = set(SAVED_CHECKPOINTS)

print("SAVED_CHECKPOINTS: ", sorted(list(SAVED_CHECKPOINTS)))

# TODO:  final_ckpt    2k, 5k, 10k   20k, 50k, 100k  150k  200k, 500k, 1M       2M, 5M, 10M