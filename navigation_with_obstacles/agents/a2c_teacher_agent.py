"""
A2C/PPO agent with an ANN-teacher -> SNN-student distillation tail.

The teacher is a FROZEN rl_games `ModelA2CContinuousLogStd.Network` (with its own
`running_mean_std`); it is fed the SAME RAW obs the student gets and normalizes internally
with its own (correct) stats, so the two never share a normalizer.
"""
import torch
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses
from navigation_with_obstacles.networks.teacher_student.teacher_builder import build_teacher


class A2CTeacherAgent(A2CAgent):
    """A2C/PPO agent that adds an ANN-teacher distillation loss to the SNN student.

    loss = ppo_loss + kd_scale(epoch) * (kd_actor_coeff * actor_kd + kd_critic_coeff * critic_kd)
    """

    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        assert self.config.get('distillation', None) is not None, \
            "Distillation config must be provided in the YAML under 'config.distillation' key."

        self.teacher_cfg = self.config.get('distillation', {})

        self.obs_dim = self.obs_shape[0]
        self.action_dim = self.actions_num

        # Distillation coefficients + anneal schedule. The tail is short by design: the
        # warm-start already put the student near the teacher, so KD only needs to hold it
        # there while PPO's critic catches up, then get out of the way.
        self.kd_actor_coeff = float(self.teacher_cfg.get('kd_actor_coeff', 0.1))
        self.kd_critic_coeff = float(self.teacher_cfg.get('kd_critic_coeff', 0.0))
        # Epochs to linearly anneal the KD scale 1 -> 0. 0 disables annealing (constant KD).
        self.kd_anneal_epochs = int(self.teacher_cfg.get('kd_anneal_epochs', 100))
        # Actor distillation divergence: 'kl' (full Gaussian, default) or 'mse' (means only,
        # matching the BC warm-up). 'kl' also pulls the student's sigma toward the teacher's.
        self.kd_actor_loss = str(self.teacher_cfg.get('kd_actor_loss', 'kl')).lower()
        assert self.kd_actor_loss in ('kl', 'mse'), \
            f"distillation.kd_actor_loss must be 'kl' or 'mse', got {self.kd_actor_loss!r}"

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
        assert self.teacher.training == False, "teacher is not frozen, check teacher builder"
        assert all(not p.requires_grad for p in self.teacher.parameters()), \
            "Teacher parameters still requires grad"

        # --- Phase 4.5: initialize the SNN critic from the ANN critic --------------------
        # Both are ANNMLPCritic with matching hidden_dims (enforced by the YAML), so weights
        # copy 1:1. 
        # The critic stays TRAINABLE:
        # a frozen teacher critic would describe the ANN's policy, not the drifting SNN's.
        self._init_critic_from_teacher()

    def _init_critic_from_teacher(self):
        """Copy the teacher's critic weights (and matching obs/value normalization stats)
        into the student. Best-effort: logs and continues if shapes/keys don't line up."""
        try:
            self.model.a2c_network.critic.load_state_dict(
                self.teacher.a2c_network.critic.state_dict())
            # Carry the normalization stats the critic was trained under, or its warm-start is
            # wasted (it must see the same input scaling, and the same value scaling under
            # normalize_value: True). PPO keeps updating these — this only seeds them.
            if self.normalize_input and hasattr(self.teacher, "running_mean_std"):
                self.model.running_mean_std.load_state_dict(
                    self.teacher.running_mean_std.state_dict())
            if self.normalize_value and hasattr(self.model, "value_mean_std") \
                    and hasattr(self.teacher, "value_mean_std"):
                self.value_mean_std.load_state_dict(
                    self.teacher.value_mean_std.state_dict())
            print("[a2c_teacher] initialized SNN critic + norm stats from the ANN teacher "
                  "(overridden by --checkpoint if one is loaded).")
        except Exception as e:  # noqa: BLE001 - never let init crash the whole run
            print(f"[a2c_teacher] WARNING: could not init critic from teacher ({e}); "
                  "critic starts from its built (random/checkpoint) weights.")

    def _current_distill_coef(self):
        """KD anneal multiplier in [0, 1]: 1 early, linearly -> 0 over `kd_anneal_epochs`,
        then 0 (the tail ends). `kd_anneal_epochs <= 0` => constant 1.0 (no anneal)."""
        if self.kd_anneal_epochs <= 0:
            return 1.0
        frac = self.epoch_num / float(self.kd_anneal_epochs)
        return max(0.0, 1.0 - frac)

    @torch.no_grad()
    def _compute_teacher_outputs(self, obs):
        """Teacher targets (mu, sigma, value) for RAW obs `obs`.

        is_train=False so the wrapper denormalizes the value into real return space (the
        teacher's value_mean_std), giving a critic-KD target comparable to the student's
        DEnormalized value. The teacher normalizes `obs` internally with its OWN frozen
        running_mean_std, so we pass the same raw obs the student sees."""
        input_dict = {
            "is_train": False,
            "obs": obs,
            "prev_actions": None,
        }
        out = self.teacher(input_dict)
        return out["mus"], out["sigmas"], out["values"]

    @staticmethod
    def _gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
        """KL( N(mu_p, sigma_p) || N(mu_q, sigma_q) ) for diagonal Gaussians, summed over
        action dims, mean over the batch. p = teacher (target), q = student.

        KL = sum_d [ log(sig_q/sig_p) + (sig_p^2 + (mu_p - mu_q)^2) / (2 sig_q^2) - 1/2 ].
        Gradients flow into the STUDENT's mu_q and sigma_q only (teacher args are detached)."""
        eps = 1e-8
        var_q = sigma_q.pow(2) + eps
        kl = (torch.log((sigma_q + eps) / (sigma_p + eps))
              + (sigma_p.pow(2) + (mu_p - mu_q).pow(2)) / (2.0 * var_q)
              - 0.5)
        return kl.sum(dim=-1).mean()

    def _actor_kd_loss(self, student_mu, student_sigma, teacher_mu, teacher_sigma):
        if self.kd_actor_loss == 'mse':
            # Mean-only BC target, gradient-equivalent to the warm-up's MSE.
            return torch.nn.functional.mse_loss(student_mu, teacher_mu)
        # Full Gaussian KL(teacher || student): also pulls student sigma toward teacher.
        return self._gaussian_kl(teacher_mu, teacher_sigma, student_mu, student_sigma)

    def calc_gradients(self, input_dict):
        """PPO gradient step + annealed teacher distillation.

        Re-implements A2CAgent.calc_gradients (rl_games a2c_continuous) and injects the KD
        term into the SAME backward, so the actor/critic see one combined gradient. Kept
        structurally identical to the base so it tracks rl_games' loss/diagnostics exactly;
        the only additions are the teacher forward and the two KD terms.
        """
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']

        # KD coefficients for THIS minibatch (annealed by epoch). Skip the teacher forward
        # entirely once the tail has fully annealed / both coeffs are zero.
        kd_scale = self._current_distill_coef()
        kd_a_coeff = kd_scale * self.kd_actor_coeff
        kd_c_coeff = kd_scale * self.kd_critic_coeff
        kd_active = (kd_a_coeff > 0.0) or (kd_c_coeff > 0.0)

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            # --- Distillation tail ------------------------------------------------------
            actor_kd = torch.zeros((), device=self.ppo_device)
            critic_kd = torch.zeros((), device=self.ppo_device)
            if kd_active:
                teacher_mu, teacher_sigma, teacher_value = self._compute_teacher_outputs(obs_batch)
                if kd_a_coeff > 0.0:
                    actor_kd = self._actor_kd_loss(mu, sigma, teacher_mu, teacher_sigma)
                    loss = loss + kd_a_coeff * actor_kd
                if kd_c_coeff > 0.0:
                    # Compare in REAL return space: denorm the student value head (values is in
                    # normalized space during is_train) against the teacher's denormed value.
                    student_value = self.model.denorm_value(values)
                    critic_kd = torch.nn.functional.mse_loss(student_value, teacher_value)
                    loss = loss + kd_c_coeff * critic_kd

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values': value_preds_batch,
            'returns': return_batch,
            'new_neglogp': action_log_probs,
            'old_neglogp': old_action_log_probs_batch,
            'masks': rnn_masks
        }, curr_e_clip, 0)

        # Surface KD scalars to TensorBoard/W&B (frame as x-axis, matching rl_games' logs).
        if self.writer is not None:
            self.writer.add_scalar('distill/kd_scale', kd_scale, self.frame)
            self.writer.add_scalar('distill/actor_kd', float(actor_kd), self.frame)
            self.writer.add_scalar('distill/critic_kd', float(critic_kd), self.frame)

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
