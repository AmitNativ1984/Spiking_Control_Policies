import torch.nn as nn
import torch

class CriticNetwork(nn.Module):
    """ Critic Network for PopSAN.
    
    This is a non-spiking, fully connected feedforward network.
    The inputs are the raw observations, and the output is the value estimate for the critic.
    """

    def __init__(self, obs_dim: int, critic_config: dict) -> None:
        """Initialize the CriticNetwork.
        
        Args:
            obs_dim (int): Dimension of the input observation space.
            critic_config (dict): Configuration dictionary for the critic network.
        """

        super(CriticNetwork, self).__init__()

        assert "hidden_dims" in critic_config, "critic configuration must include 'hidden_dims' key"
        hidden_dims = critic_config["hidden_dims"]

        self.critic_fc1 = nn.Linear(in_features=obs_dim, out_features=hidden_dims[0])
        self.critic_fc2 = nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1])
        self.critic_fc3 = nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[2])
        self.value_head = nn.Linear(in_features=hidden_dims[2], out_features=1)
    
        self.model = nn.Sequential(
            self.critic_fc1,
            nn.ELU(),
            self.critic_fc2,
            nn.ELU(),
            self.critic_fc3,
            nn.ELU(),
            self.value_head
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CriticNetwork.
        
        Args:
            state (torch.Tensor): Input state tensor of shape [batch_size, obs_dim].
            
        Returns:
            torch.Tensor: Value estimate tensor of shape [batch_size, 1].
        """
        value = self.model(state)
        return value