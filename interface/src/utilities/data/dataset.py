# Dataset
import csv
import numpy as np
from torch.utils.data import Dataset
import os


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup

class TextDataset(Dataset):
    def __init__(self, data, logfile):
        super().__init__()
        self.data = data
        self.logfile = logfile
    def __getitem__(self, index):
        data_dict = {}
         # construct dict
        data_dict['fname'] = f"infer_file_{index}"
        data_dict['fbank'] = np.zeros((1024,64))
        data_dict['waveform'] = np.zeros((32000))
        data_dict['text'] = self.data[index]
        if index == 0:
            with open(os.path.join(self.logfile), 'w') as f:
                f.write(f"{data_dict['fname']}: {data_dict['text']}")
        else:
            with open(os.path.join(self.logfile), 'a') as f:
                f.write(f"\n{data_dict['fname']}: {data_dict['text']}")
        return data_dict


    def __len__(self):
        return len(self.data)

