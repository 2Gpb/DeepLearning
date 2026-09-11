import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.rnn = rnn_type(embed_size, hidden_size, rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        output = self.embedding(indices)
        packed = pack_padded_sequence(output, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.linear(output)

        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        assert temp > 0
        self.eval()

        encoded = self.dataset.sp_model.encode(prefix)
        indices = [self.dataset.bos_id] + encoded[:self.max_length - 2]
        
        device = next(self.parameters()).device
        input_indices = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)

        output = self.embedding(input_indices)
        output, h_state = self.rnn(output)
        logits = self.linear(output[:, -1, :])

        distribution = torch.distributions.Categorical(logits=logits / temp)
        new_token = distribution.sample()

        indices.append(new_token.item())

        while len(indices) < self.max_length and indices[-1] != self.dataset.eos_id:
            new_token_input = new_token.unsqueeze(1)

            new_embedding = self.embedding(new_token_input)
            output, h_state = self.rnn(new_embedding, h_state)
            logits = self.linear(output[:, -1, :])

            distribution = torch.distributions.Categorical(logits=logits / temp)
            new_token = distribution.sample()

            indices.append(new_token.item())

        generated = self.dataset.sp_model.decode(indices)
        return generated
