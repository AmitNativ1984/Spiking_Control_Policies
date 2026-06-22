import torch
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch import model_builder
from rl_games.algos_torch import torch_ext
from networks.ann.actor_critic import MLPActorCriticNetworkBuilder


class A2CTeacherAgent(A2CAgent):
    """A2C/PPO agent that add an ANN-teacher distillation loss the the SNN student
    
    loss = ppo_loss + distil_coef(epoch) * distillation_loss(teacher, student)
    """
    
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        assert self.config.get('distillation', None) is not None, "Distillation config must be provided in the YAML under 'config.distillation' key."
        
        self.teacher_params = self.config.get('distillation',{})

        self.obs_dim = self.obs_shape[0]
        self.action_dim = self.actions_num

        self.distill_coef = self.teacher_params.get('distill_coef', 1.0)
        
        self._build_teacher()  # Build the teacher network
        checkpoint = torch.load(self.teacher_params.get('checkpoint', ''))
        self.teacher_network.load_state_dict(checkpoint['model_state_dict'])
        self.teacher_model = self.teacher_network.to(params['device'])

        # Set the teacher model to evaluation mode and freeze its parameters
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def _build_teacher(self):
        """Build the teacher network based on the provided configuration."""
        
        self.teacher_network = MLPActorCriticNetworkBuilder.build(
            name=self.teacher_params.get('name', 'MLPActorCriticNetwork'),
            input_shape=self.obs_dim,
            actions_num=self.action_dim,
            **self.teacher_params["network"]
        )
        return
    
    def _current_distill_coef(self):
        """Return the current distillation coefficient, which may be annealed over epochs."""
        raise NotImplementedError("Implement distillation coefficient annealing logic here if needed.")
        
    @torch.no_grad()
    def _compute_teacher_outputs(self, obs):
        """Compute the teacher's outputs (mu, sigma) given the observations."""
        raise NotImplementedError("Implement the logic to compute teacher outputs here.")

    def calc_gradients(self, input_dict):
        return super().calc_gradients(input_dict)        