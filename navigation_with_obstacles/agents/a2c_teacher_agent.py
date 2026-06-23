import torch
from rl_games.algos_torch.a2c_continuous import A2CAgent
from navigation_with_obstacles.networks.teacher_student.teacher_builder import build_teacher


class A2CTeacherAgent(A2CAgent):
    """A2C/PPO agent that add an ANN-teacher distillation loss the the SNN student
    
    loss = ppo_loss + distil_coef(epoch) * distillation_loss(teacher, student)
    """
    
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        assert self.config.get('distillation', None) is not None, "Distillation config must be provided in the YAML under 'config.distillation' key."
        
        self.teacher_cfg = self.config.get('distillation',{})

        self.obs_dim = self.obs_shape[0]
        self.action_dim = self.actions_num

        self.distill_coef = self.teacher_cfg.get('distill_coef', 1.0)
        
        # load a FROZEN teacher
        self.teacher = build_teacher(
            teacher_network_cfg=self.teacher_cfg["network"],
            model_name=params["model"]["name"],
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            checkpoint_path=self.teacher_cfg["checkpoint"],
            device=self.ppo_device,
            normalize_input=self.teacher_cfg["normalize_input"],
            normalize_value=self.teacher_cfg["normalize_value"],
        )
        assert self.teacher.training == False, (f"teacher is not frozen, check teacher builder")
        assert all(not p.requires_grad for p in self.teacher.parameters()), "Teacher parameters still requires grad"


    def _current_distill_coef(self):
        """Return the current distillation coefficient, which may be annealed over epochs."""
        raise NotImplementedError("Implement distillation coefficient annealing logic here if needed.")
        
    @torch.no_grad()
    def _compute_teacher_outputs(self, obs):
        """Compute the teacher's outputs (mu, sigma) given the observations."""
        input_dict = {
            "is_train": False,
            "obs": obs,
            "prev_actions": None           
        }
        
        action = self.teacher(input_dict)
        return action["mus"], action["sigmas"]

    def calc_gradients(self, input_dict):
        return super().calc_gradients(input_dict)        