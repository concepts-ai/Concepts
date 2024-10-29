#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vae1d.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import jactorch.nn as jacnn

from typing import Optional, List, Any, TypeVar, Tuple

Tensor = TypeVar('torch.Tensor')


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    training: bool

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu: Mean of the latent Gaussian [B x D]
            logvar: Standard deviation of the latent Gaussian [B x D]

        Returns:
            z: Samples from the latent Gaussian [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input, **kwargs)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z, **kwargs), input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, **kwargs) -> dict:
        kld_weight = 1  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1))
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'loss/recon':recons_loss.item(), 'loss/kl':-kld_loss.item()}


class VanillaVAE1d(BaseVAE):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Optional[List[int]] = None, **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.encoder = jacnn.MLPLayer(input_dim, latent_dim * 2, hidden_dims, activation=nn.LeakyReLU(), flatten=False, last_activation=False)
        self.decoder = jacnn.MLPLayer(latent_dim, input_dim, list(reversed(hidden_dims)), activation=nn.LeakyReLU(), flatten=False, last_activation=False)

    training: bool

    input_dim: int
    """The dimension of the input data."""

    latent_dim: int
    """The dimension of the latent space."""

    hidden_dims: Optional[List[int]]
    """The hidden dimensions of the encoder and decoder. If None, the encoder and decoder are single-layer networks."""

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        mu, log_var = result.split(self.latent_dim, dim=-1)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def sample(self, nr_samples:int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(nr_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples


class ConditionalVAE1d(BaseVAE):
    def __init__(self, input_dim: int, condition_dim: int, latent_dim: int, hidden_dims: Optional[List[int]] = None, **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.encoder = jacnn.MLPLayer(input_dim + condition_dim, latent_dim * 2, hidden_dims, activation=nn.LeakyReLU(), flatten=False, last_activation=False)
        self.decoder = jacnn.MLPLayer(latent_dim + condition_dim, input_dim, list(reversed(hidden_dims)), activation=nn.LeakyReLU(), flatten=False, last_activation=False)

    training: bool

    input_dim: int
    """The dimension of the input data."""

    condition_dim: int
    """The dimension of the condition."""

    latent_dim: int
    """The dimension of the latent space."""

    hidden_dims: Optional[List[int]]
    """The hidden dimensions of the encoder and decoder. If None, the encoder and decoder are single-layer networks."""

    def encode(self, input: Tensor, label: Tensor) -> List[Tensor]:
        result = self.encoder(torch.cat([input, label], dim=-1))
        mu, log_var = result.split(self.latent_dim, dim=-1)
        return [mu, log_var]

    def decode(self, z: Tensor, label: Tensor) -> Tensor:
        return self.decoder(torch.cat([z, label], dim=-1))

    def sample(self, label: Tensor, nr_samples: int, **kwargs) -> Tensor:
        z = torch.randn(nr_samples, self.latent_dim)
        label = label.unsqueeze(0).expand((nr_samples, label.size(0)))
        samples = self.decode(z, label)
        return samples

