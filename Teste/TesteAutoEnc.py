from Autoencoder.Rede import Encoder, Decoder
import torch
from Autoencoder.LoaderDados import Funcoes
import torchaudio
import matplotlib.pyplot as plt


# carregando dados para teste
functions = Funcoes()
path = r"F:\Projetos\Autoencoder/"
sinal_name = "dis-f1-b1_noise.wav"
signal, sr = torchaudio.load(path + sinal_name)
print(signal.max(), signal.min())
signal = functions.resample_if_necessary(signal, sr)
signal = functions.mix_down_if_necessary(signal)
signal = functions.cut_if_necessary(signal)
signal = functions.right_pad_if_necessary(signal)
plt.plot(signal[0].cpu().numpy(), label="original")


encoder = Encoder(100)
decoder = Decoder(100)

# carregando o modelo salvo
encoder.load_state_dict(torch.load('../modelos/Segundo_tanh/encoderRede_basicona_tanh.pth'))
decoder.load_state_dict(torch.load('../modelos/Segundo_tanh/decoderRede_basicona_tanh.pth'))
print("Shape:", signal.shape)
saida = encoder(signal.reshape(1, 1, -1))
saida = decoder(saida)

plt.plot(saida[0][0].cpu().detach().numpy(), label="saida")

print(saida.max(), saida.min())
plt.legend()
plt.show()
# salvando o arquivo
torchaudio.save(f'reconstrucao_{sinal_name}.wav', saida.reshape(1, -1).detach(), 16000)
