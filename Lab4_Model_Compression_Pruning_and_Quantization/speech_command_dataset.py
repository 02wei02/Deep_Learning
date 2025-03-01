import os
import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np


class SpeechCommandDataset(Dataset):
    def __init__(self, data_path='/content/drive/MyDrive/DL_lab4/speech_commands/', is_training=True, transform=None):
        self.data_path = data_path
        self.is_training = is_training
        self.transform = transform

        if is_training:
            self.data_list_path = os.path.join(
                self.data_path, 'train_list.txt')
        else:
            self.data_list_path = os.path.join(self.data_path, 'test_list.txt')

        self.ids = [id.strip() for id in open(self.data_list_path)]
        # self.classes = classes = ['yes', 'no', 'up', 'down', 'left',
        #                           'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']
        self.classes = classes = ['yes', 'no', 'up', 'down', 'left',
                                  'right', 'on', 'off', 'stop', 'go']
        self.num_classes = len(self.classes)
        self.num_audios = len(self.ids)

    def __len__(self):
        return self.num_audios

    def __getitem__(self, index):
        id = self.ids[index]

        audio, sr = sf.read(os.path.join(self.data_path, id))
        audio = audio.astype(np.float32)

        if self.transform is not None:
            audio = self.transform(audio)

        # padding or truncating to 1 second
        if len(audio) < sr:
            padding_size = (sr-len(audio))//2
            if len(audio) % 2 == 0:
                audio = np.pad(audio, (padding_size, padding_size),
                               mode='constant')
            else:
                audio = np.pad(audio, (padding_size, padding_size+1),
                               mode='constant')
        elif len(audio) > sr:
            truncating_size = (len(audio)-sr)//2
            if len(audio) % 2 == 0:
                audio = audio[truncating_size:truncating_size+sr]
            else:
                audio = audio[truncating_size+1:truncating_size+1+sr]

        audio = audio[np.newaxis, ...]

        label = self.classes.index(os.path.split(id)[0])
        # label = int(self.classes.index(os.path.split(id)[0]))
        label = np.array(label, dtype=np.int)
        # label = int(self.classes.index(os.path.split(id)[0]))  # Direct integer assignment without np.array
        label = torch.tensor(self.classes.index(os.path.split(id)[0]), dtype=torch.long)

        return audio, label
