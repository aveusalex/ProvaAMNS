from torch.utils.data import Dataset
import torchaudio
import torch
import pandas as pd


class DadosAE(Dataset):
    def __init__(self,
               target_sample_rate):
        self.raiz = r"F:\Projetos\Autoencoder"
        self.annotations = pd.read_csv(r"F:\Projetos\Autoencoder\metadata\treino.csv")
        self.annotations_ruido = pd.read_csv(r"F:\Projetos\Autoencoder\metadata\treinoruidoso.csv")
        self.device = "cuda"
        self.target_sample_rate = target_sample_rate
        self.num_samples = 4 * 16000
        self.cache = dict()  # controlamos para ter no máximo 100 amostras no cache

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        path_ruido, path_limpo = self._get_audio_sample_path(index)

        # audio com ruido (X)
        signal, sr = torchaudio.load(self.raiz + "/" + path_ruido)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        # audio limpo (y)
        signal_limpo, sr_limpo = torchaudio.load(self.raiz + "/" + path_limpo)
        signal_limpo = signal_limpo.to(self.device)
        signal_limpo = self._resample_if_necessary(signal_limpo, sr_limpo)
        signal_limpo = self._mix_down_if_necessary(signal_limpo)
        signal_limpo = self._cut_if_necessary(signal_limpo)
        signal_limpo = self._right_pad_if_necessary(signal_limpo)

        if len(self.cache) < 256:
            self.cache[index] = (signal, signal_limpo)
        else:
            self.cache = dict()
            self.cache[index] = (signal, signal_limpo)

        return signal, signal_limpo

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler.to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        return self.annotations_ruido.iloc[index, 0], self.annotations.iloc[index, 0]

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]


class Funcoes:
    def __init__(self, target_sample_rate=16000, device="cpu"):
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = 4 * 16000

    def cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler.to(self.device)
            signal = resampler(signal)
        return signal

    def mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal