# ProvaAMNS
Nesse repositório, estão localizados os códigos gerados para a segunda prova de Aprendizado Não Supervisionado.
A resolução da prova está no PDF armazenado na raiz do repositório.

## Da organização do repositório:

O repositório está dividido em pastas que contém os módulos criados para cada frente de desenvolvimento.
- Autoencoder: essa pasta contém os códigos que carregam os dados utilizados para o treinamento do autoencoder, a declaração da rede neural e o loop de treinamento.
- CNN_classificacao: essa pasta contém notebooks referentes ao processo de colocar ruído nos áudios e de treinamento da rede neural de classficação de sentimentos em voz.
- JuncaoAutoCNN: essa pasta contém os notebooks de junção das duas redes neurais, o AutoEncoder e a CNN de classificação de sentimento, sendo um notebook para cada modelo de autoencoder.
- Teste: nessa pasta está um script utilziado para testar o autoencoder e seu funcionamento. Ele lê áudios presentes na subpasta Sinais e passa pelo autoencoder, gravando a saída e gerando um gráfico de comparação do sinal original e do sinal gerado.
- Modelos: essa pasta contém os modelos treinados do autoencoder.


> Mais detalhes estão no PDF anexo.
