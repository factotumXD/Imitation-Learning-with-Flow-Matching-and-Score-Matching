"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn

import math

class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ### Update: done
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        current_size = self.state_dim
        for size in hidden_dims:
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.SiLU())
            current_size = size
        layers.append(nn.Linear(current_size, self.chunk_size*self.action_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mean(
            torch.linalg.norm(
                action_chunk - self.sample_actions(state),
                dim=(-1,-2)
            )**2
        )

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.mlp(state).view(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ### Update: done
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        current_size = self.state_dim + self.chunk_size*self.action_dim + 1
        for size in hidden_dims:
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.SiLU())
            current_size = size
        layers.append(nn.Linear(current_size, self.chunk_size*self.action_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        a_0 = torch.randn_like(action_chunk)
        tau = torch.rand(B, 1, device=state.device)
        a_tau = tau*action_chunk.view(B,-1) + (1-tau)*a_0.view(B,-1)
        
        input = torch.concat([state, a_tau.view(B,-1), tau], dim=-1)
        v = self.mlp(input).view(-1, self.chunk_size, self.action_dim)

        return torch.mean(
            torch.linalg.norm(
                v - (action_chunk - a_0),
                dim=(-1,-2)
            )**2
        )

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        action_chunk = torch.randn(B, self.chunk_size, self.action_dim)
        for i in range(num_steps):
            tau = torch.full((B,1), i / num_steps, device=state.device)
            input = torch.concat([state, action_chunk.view(B,-1), tau.expand(B,1)], dim=-1)
            v = self.mlp(input).view(-1, self.chunk_size, self.action_dim)
            action_chunk = action_chunk + v / num_steps
        return action_chunk

class ScoreMatchingPolicy(BasePolicy):
    """Predicts action chunks with a denoising score matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        sigma: float = 1
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.sigma=sigma

        layers = []
        current_size = self.state_dim + self.chunk_size * self.action_dim + 1
        for size in hidden_dims:
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.SiLU())
            current_size = size
        layers.append(nn.Linear(current_size, self.chunk_size * self.action_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        a_0 = torch.randn_like(action_chunk)
        tau = torch.rand(B, 1, device=state.device)
        a_tau = tau*action_chunk.view(B,-1) + (1-tau)*a_0.view(B,-1)
        
        input = torch.concat([state, a_tau.view(B,-1), tau], dim=-1)
        v = self.mlp(input).view(-1, self.chunk_size, self.action_dim)

        return torch.mean(
            torch.linalg.norm(
                v - (action_chunk - a_0),
                dim=(-1,-2)
            )**2
        )

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        action_chunk = torch.randn(B, self.chunk_size, self.action_dim)
        step_size=1.0/num_steps
        for i in range(num_steps):
            t=i/num_steps
            tau = torch.full((B,1), i / num_steps, device=state.device)
            input = torch.concat([state, action_chunk.view(B,-1), tau.expand(B,1)], dim=-1)
            v = self.mlp(input).view(-1, self.chunk_size, self.action_dim)
            score=(t*v-action_chunk)/(1-t)
            update=(v+0.5*self.sigma**2*score)*step_size+self.sigma * math.sqrt(step_size)*torch.randn(B, self.chunk_size, self.action_dim)
            action_chunk = action_chunk + update
        return action_chunk



PolicyType: TypeAlias = Literal["mse", "flow","score"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "score":
        return ScoreMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
