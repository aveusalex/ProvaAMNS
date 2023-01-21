import torch
from torch.utils.data import DataLoader, random_split
# meus codigos
from LoaderDados import DadosAE
from Rede import Encoder, Decoder
import matplotlib.pyplot as plt
from math import floor


# hiperparametros
batch_size = 256
num_epochs = 10
learning_rate = 1e-5
encoded_space_dim = 100


# Define o modelo, otimizador e perda
encoder = Encoder(encoded_space_dim)
decoder = Decoder(encoded_space_dim)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate)
loss_fn = torch.nn.MSELoss()

# movendo para a GPU
encoder = encoder.cuda()
decoder = decoder.cuda()
loss_fn = loss_fn.cuda()

# Carregue seus dados de treinamento
train_data = DadosAE(target_sample_rate=16000)

train_size = int(0.8 * len(train_data))  # dividindo em treino e validação
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Armazenando as losses para verificar o overfitting
train_losses = []
val_losses = []

# Loop de treinamento
# Loop de épocas
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    # Treine o modelo com o subset de treinamento
    encoder.train()
    decoder.train()

    train_loss = 0
    step = 0
    for input, target in train_loader:
        optimizer.zero_grad()
        encoded_Data = encoder(input)
        decoded_Data = decoder(encoded_Data)
        loss = loss_fn(decoded_Data, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if step % 10 == 0:
            progress = step / len(train_loader)
            # Calculate the number of hashes to represent the progress
            hashes = '#' * int(floor(progress * 50))
            spaces = ' ' * (50 - len(hashes))
            print(f'\rStep {step} [{hashes}{spaces}] {progress:.2%} /{len(train_loader)} - Loss: {loss.item():.5f}', end='')
        step += 1
    train_loss /= len(train_loader)  # calculando a loss media dos steps / batches
    train_losses.append(train_loss)

    print("\n")

    # Avalie o modelo com o subset de validação
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        val_loss = 0
        for input, target in val_loader:
            encoded_Data = encoder(input)
            decoded_Data = decoder(encoded_Data)
            val_loss += loss_fn(decoded_Data, target).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Validation Loss: {val_loss}\n')

# Salve o modelo
torch.save(encoder.state_dict(), 'encoder.pth')
torch.save(decoder.state_dict(), 'decoder.pth')

# Plotar as losses
plt.figure(figsize=(16,9))
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.title('Losses')
plt.show()
