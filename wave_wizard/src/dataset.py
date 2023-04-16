import os

import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_train_val_test_loaders(config):
    train_set = AudioSet(
        file_path=config["train"]["ann_path"],
        sample_rate=config["sample_rate"],
        sample_len=config["sample_len"],
        mode="train",
        shift=config["shift"],
    )
    train_loader = DataLoader(train_set, **config["train"]["dataloader"])

    if "val" in config:
        val_set = AudioSet(
            file_path=config["train"]["ann_path"],
            sample_rate=config["sample_rate"],
            sample_len=config["sample_len"],
            mode="train",
            shift=config["shift"],
        )
        val_loader = DataLoader(val_set, **config["val"]["dataloader"])
    else:
        val_loader = None

    if "test" in config:
        test_set = AudioSet(
            file_path=config["test"]["ann_path"],
            sample_rate=config["sample_rate"],
            sample_len=config["sample_len"],
            mode="test",
            shift=config["shift"],
        )
        test_loader = DataLoader(test_set, **config["test"]["dataloader"])
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def sample_fixed_length_data_aligned(data_a, data_b, sample_len):
    """sample with fixed length from two dataset"""
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert (
        len(data_a) >= sample_len
    ), f"len(data_a) is {len(data_a)}, sample_len is {sample_len}."

    frames_total = len(data_a)
    start = np.random.randint(frames_total - sample_len + 1)
    end = start + sample_len

    return data_a[start:end], data_b[start:end]


class AudioSet(Dataset):
    def __init__(
        self,
        file_path,
        sample_rate=1600,
        sample_len=16384,
        mode="train",
        shift=0,
    ):
        super(Dataset, self).__init__()
        dataset_list = [
            line.rstrip("\n").rstrip()
            for line in open(os.path.abspath(os.path.expanduser(file_path)), "r")
        ]

        assert mode in ("train", "val", "test")
        self.sr = sample_rate
        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_len = sample_len + shift
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        noisy_path, clean_path = self.dataset_list[i].split(" ")
        filename = os.path.splitext(os.path.basename(noisy_path))[0]
        noisy, _ = librosa.load(
            os.path.abspath(os.path.expanduser(noisy_path)),
            sr=self.sr,
        )

        clean, _ = librosa.load(
            os.path.abspath(os.path.expanduser(clean_path)),
            sr=self.sr,
        )

        assert len(clean) == len(noisy)

        if self.mode == "train":
            # The input of model should be fixed-length in the training.
            noisy, clean = sample_fixed_length_data_aligned(
                noisy, clean, self.sample_len
            )
            return noisy.reshape(1, -1), clean.reshape(1, -1), filename
        else:
            return noisy.reshape(1, -1), clean.reshape(1, -1), filename
