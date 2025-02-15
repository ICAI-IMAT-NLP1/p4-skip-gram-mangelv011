import torch
from torch import nn
import torch.optim as optim
device = "cuda" if torch.cuda.is_available() else "cpu"
class SkipGramNeg(nn.Module):
    """A SkipGram model with Negative Sampling.

    This module implements a SkipGram model using negative sampling. It includes
    embedding layers for input and output words and initializes these embeddings
    with a uniform distribution to aid in convergence.

    Attributes:
        n_vocab: An integer count of the vocabulary size.
        n_embed: An integer specifying the dimensionality of the embeddings.
        noise_dist: A tensor representing the distribution of noise words.
        in_embed: The embedding layer for input words.
        out_embed: The embedding layer for output words.
    """

    def __init__(self, n_vocab: int, n_embed: int, noise_dist: torch.Tensor = None):
        """Initializes the SkipGramNeg model with given vocabulary size, embedding size, and noise distribution.

        Args:
            n_vocab: The size of the vocabulary.
            n_embed: The size of each embedding vector.
            noise_dist: The distribution of noise words for negative sampling.
        """
        super().__init__()
        self.n_vocab: int = n_vocab
        self.n_embed: int = n_embed
        self.noise_dist: torch.Tensor = noise_dist

        # Define embedding layers for input and output words
        # TODO
        self.in_embed: nn.Embedding = nn.Embedding(n_vocab, n_embed).to(device)
        self.out_embed: nn.Embedding = nn.Embedding(n_vocab, n_embed).to(device)

        # Initialize embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

        self.to(device)


    def forward_input(self, input_words: torch.Tensor) -> torch.Tensor:
        """Fetches input vectors for a batch of input words.

        Args:
            input_words: A tensor of integers representing input words.

        Returns:
            A tensor containing the input vectors for the given words.
        """
        # TODO
        input_words = input_words.to(device)
        input_vectors: torch.Tensor = self.in_embed(input_words)
        return input_vectors

    def forward_output(self, output_words: torch.Tensor) -> torch.Tensor:
        """Fetches output vectors for a batch of output words.

        Args:
            output_words: A tensor of integers representing output words.

        Returns:
            A tensor containing the output vectors for the given words.
        """
        # TODO
        output_words = output_words.to(device)
        output_vectors: torch.Tensor = self.out_embed(output_words)
        return output_vectors

    def forward_noise(self, batch_size: int, n_samples: int) -> torch.Tensor:
        """Generates noise vectors for negative sampling.

        Args:
            batch_size: The number of words in each batch.
            n_samples: The number of negative samples to generate per word.

        Returns:
            A tensor of noise vectors with shape (batch_size, n_samples, n_embed).
        """
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist: torch.Tensor = torch.ones(self.n_vocab)
        else:
            noise_dist: torch.Tensor = self.noise_dist
        noise_dist = noise_dist.to(device)
        # Sample words from our noise distribution
        # TODO
        noise_words: torch.Tensor =  torch.multinomial(noise_dist,
                                      batch_size * n_samples,
                                      replacement=True)


     
        # Reshape output vectors to size (batch_size, n_samples, n_embed)
        # TODO
        noise_vectors: torch.Tensor = self.out_embed(noise_words).view(
            batch_size, n_samples, self.n_embed)


        return noise_vectors

    
class NegativeSamplingLoss(nn.Module):
    """Implements the Negative Sampling loss as a PyTorch module.

    This loss is used for training word embedding models like Word2Vec using
    negative sampling. It computes the loss as the sum of the log-sigmoid of
    the dot product of input and output vectors (for positive samples) and the
    log-sigmoid of the dot product of input vectors and noise vectors (for
    negative samples), across a batch.
    """

    def __init__(self):
        """Initializes the NegativeSamplingLoss module."""
        super().__init__()

    def forward(self, input_vectors: torch.Tensor, output_vectors: torch.Tensor,
                noise_vectors: torch.Tensor) -> torch.Tensor:
        """Computes the Negative Sampling loss.

        Args:
            input_vectors: A tensor containing input word vectors, 
                            shape (batch_size, embed_size).
            output_vectors: A tensor containing output word vectors (positive samples), 
                            shape (batch_size, embed_size).
            noise_vectors: A tensor containing vectors for negative samples, 
                            shape (batch_size, n_samples, embed_size).

        Returns:
            A tensor containing the average loss for the batch.
        """
        batch_size, n_embed = input_vectors.shape
        
        # Compute log-sigmoid loss for correct classifications
        # Reshape input vectors to match output vectors for batch dot product
        input_vectors = input_vectors.view(batch_size, n_embed, 1)
        # Reshape output vectors for batch dot product
        output_vectors = output_vectors.view(batch_size, 1, n_embed)
        # For positive samples: -log(sigmoid(x))
        out_loss = -torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()

        # Compute log-sigmoid loss for incorrect classifications
        input_vectors = input_vectors.view(batch_size, n_embed, 1)
        # Compute dot product between input vectors and all noise vectors
        noise_dot = torch.bmm(noise_vectors, input_vectors)
        # For negative samples: -log(1 - sigmoid(x)) = -log(sigmoid(-x))
        noise_loss = -noise_dot.neg().sigmoid().log()
        noise_loss = noise_loss.squeeze()

        # Return the sum of both losses (now both positive), averaged over the batch
        return (out_loss.sum() + noise_loss.sum()) / batch_size