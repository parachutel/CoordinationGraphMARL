import collections
import copy

from dowel import tabular, logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from garage import log_performance, TrajectoryBatch
from garage.misc import tensor_utils
from garage.torch.algos import (_Default, filter_valids,
                                make_optimizer, pad_to_last)
from garage.torch.utils import flatten_batch

# from dicg.np.algos import MABatchPolopt
from dicg.torch.algos.utils import pad_one_to_last

class CentralizedMAPPO(nn.Module):

    def __init__(
            self,
            policy,
            baseline,
            # edge_predictor=None,
            optimizer=torch.optim.Adam,
            baseline_optimizer=torch.optim.Adam,
            optimization_n_minibatches=1,
            optimization_mini_epochs=1,
            policy_lr=_Default(3e-4),
            baseline_lr=_Default(3e-4),
            lr_clip_range=2e-1,
            max_path_length=500,
            num_train_per_epoch=1,
            discount=0.99,
            gae_lambda=1,
            center_adv=True,
            positive_adv=False,
            policy_ent_coeff=0.0,
            use_softplus_entropy=False,
            stop_entropy_gradient=False,
            entropy_method='no_entropy',
            clip_grad_norm=None,
            device='cpu'
    ):

        super().__init__()

        self.policy = policy
        self.baseline = baseline
        # if policy.proximity_adj:
        #     self.edge_predictor = edge_predictor
        self.discount = discount
        self.max_path_length = max_path_length
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._lr_clip_range = lr_clip_range
        self._eps = 1e-8

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=100)

        self._optimizer = make_optimizer(optimizer,
                                         policy,
                                         lr=policy_lr,
                                         eps=_Default(1e-5))

        self._baseline_optimizer = make_optimizer(baseline_optimizer,
                                                  baseline,
                                                  lr=baseline_lr,
                                                  eps=_Default(1e-5))
        
        # if policy.proximity_adj:
        #     self._edge_predictor_optimizer = make_optimizer(optimizer, 
        #                                                     edge_predictor,
        #                                                     lr=6e-5,
        #                                                     eps=_Default(1e-5))

        self._optimization_n_minibatches = optimization_n_minibatches
        self._optimization_mini_epochs = optimization_mini_epochs

        self._clip_grad_norm = clip_grad_norm

        self._old_policy = copy.deepcopy(self.policy)

        self.device = device

    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    def train_once(self, itr, paths, tbx):
        logger.log('Processing samples...')
        obs, avail_actions, actions, rewards, valids, returns, \
            adjs, alive_masks = self.process_paths(paths)

        # print('processed obs.shape =', obs.shape)
        # print('processed avail_actions.shape=', avail_actions.shape)
        # print(avail_actions)

        # with torch.no_grad():
        #     loss_before = self._compute_loss(itr, obs, avail_actions, actions, 
        #                                      rewards, valids, baselines, adjs,
        #                                      alive_masks)

        self._old_policy.load_state_dict(self.policy.state_dict())

        # Start train with path-shuffling
        policy_grad_norms = []
        policy_entropies = []
        step_size = int(np.ceil(len(rewards) / self._optimization_n_minibatches))

        # step_size = int(self._minibatch_size / self.policy._n_agents) \
        #     if self._minibatch_size else len(rewards)

        shuffled_ids = np.random.permutation(len(rewards))
        # shuffled_ids = np.array(range(len(rewards)))
        print('MultiAgentNumTrajs =', len(rewards))

        for mini_epoch in range(self._optimization_mini_epochs):
            for start in range(0, len(rewards), step_size):
                ids = shuffled_ids[start : min(start + step_size, len(rewards))]
                logger.log('Mini epoch: {} | Optimizing policy using traj {} to traj {}'.
                    format(mini_epoch, start, min(start + step_size, len(rewards)))
                )
                if not self.policy.proximity_adj:
                    # e.g. DICG
                    adjs_ids = None
                    alive_masks_ids = alive_masks[ids]
                else:
                    adjs_ids = adjs[ids] # Notice naming
                    alive_masks_ids = None

                obs_ids = obs[ids].to(self.device)
                avail_actions_ids = avail_actions[ids]
                actions_ids = actions[ids].to(self.device)
                rewards_ids = rewards[ids].to(self.device)
                returns_ids = returns[ids].to(self.device)
                valids_ids = valids[ids].to(self.device)
                # baselines_ids = baselines[ids].to(self.device)
                adjs_ids = adjs_ids.to(self.device) if adjs is not None else None
                alive_masks_ids = alive_masks_ids.to(self.device) if alive_masks is not None else None

                with torch.no_grad():
                    baselines_ids = self.baseline.forward(obs_ids)

                loss = self._compute_loss(itr, obs_ids, avail_actions_ids, 
                                          actions_ids, rewards_ids, 
                                          valids_ids, baselines_ids, adjs_ids,
                                          alive_masks_ids)
                self._optimizer.zero_grad()
                loss.backward()

                # if self.policy.proximity_adj:
                #     edge_predictor_loss = self.edge_predictor.get_loss(
                #         obs[ids], adjs_ids, valids[ids])
                #     self._edge_predictor_optimizer.zero_grad()
                #     edge_predictor_loss.backward()

                baseline_loss = self.baseline.compute_loss(
                    obs_ids, returns_ids, valids_ids)
                self._baseline_optimizer.zero_grad()
                baseline_loss.backward()
    
                if self._clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                                   self._clip_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.baseline.parameters(), 
                                                   self._clip_grad_norm)
                policy_grad_norms.append(self.policy.grad_norm())

                self._optimizer.step()
                self._baseline_optimizer.step()

                with torch.no_grad():
                    policy_entropy = self._compute_policy_entropy(obs_ids, avail_actions_ids, 
                        actions_ids, adjs=adjs_ids, alive_masks=alive_masks_ids)
                    policy_entropies.append(policy_entropy.mean().item())
                
                if self.device != 'cpu':
                    torch.cuda.empty_cache()

            logger.log('Mini epoch: {} | Loss: {:.5f}'.format(mini_epoch, loss.item()))
            logger.log('Mini epoch: {} | BaselineLoss: {:.5f}'.format(
                mini_epoch, baseline_loss.item()))
            # if self.policy.proximity_adj:
            #     logger.log('Mini epoch: {} | EdgePredictorLoss: {:.5f}'.format(
            #         mini_epoch, edge_predictor_loss.item()))

        # End train

        # with torch.no_grad():
        #     # loss_after = self._compute_loss(itr, obs, avail_actions, actions, 
        #     #                                 rewards, valids, baselines, adjs)
        #     policy_entropy = self._compute_policy_entropy(obs, avail_actions, 
        #         actions, adjs=adjs, alive_masks=alive_masks)


        # logging ##############################################################
        # log_performance customization block
        n_agents = actions.shape[-1]
        returns = []
        undiscounted_returns = []
        for i_path in range(len(paths)):
            path_rewards = np.asarray(paths[i_path]['rewards'])
            returns.append(paths[i_path]['returns'])
            undiscounted_returns.append(np.sum(path_rewards))

        average_discounted_return = np.mean([r[0] for r in returns])
    
        tabular.record('Iteration', itr)
        tabular.record('NumTrajs', len(paths))
        # tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        # tabular.record('LossBefore', loss.item())
        # tabular.record('LossAfter', loss_after.item())
        # tabular.record('dLoss', loss.item() - loss_after.item())
        tabular.record('Entropy', np.mean(policy_entropies))
        tabular.record('GradNorm', np.mean(policy_grad_norms))
        tabular.record('BaselineLoss', baseline_loss.item())

        tbx.add_scalar('train/avg_return', np.mean(undiscounted_returns), itr)
        tbx.add_scalar('train/max_return', np.max(undiscounted_returns), itr)
        tbx.add_scalar('train/min_return', np.min(undiscounted_returns), itr)
        tbx.add_scalar('train/entropy', np.mean(policy_entropies), itr)
        tbx.add_scalar('train/baseline_loss', baseline_loss.item(), itr)
        tbx.add_scalar('train/grad_norm', np.mean(policy_grad_norms), itr)
        
        # if self.policy.proximity_adj:
        #     tabular.record('EdgePredictorLoss', edge_predictor_loss.item())
        #     tbx.add_scalar('train/edge_predictor_loss', edge_predictor_loss.item(), itr)
        
        return np.mean(undiscounted_returns)
    
    def compute_advantages(self, max_path_length, baselines, rewards):
        adv_filter = torch.full((1, 1, 1, max_path_length - 1),
                                self.discount * self._gae_lambda).to(self.device)
        adv_filter = torch.cumprod(F.pad(adv_filter, (1, 0), value=1), dim=-1)
    
        deltas = (rewards + self.discount * F.pad(baselines, (0, 1))[:, 1:] - baselines)
        deltas = F.pad(deltas, (0, max_path_length - 1)).unsqueeze(0).unsqueeze(0)
    
        advantages = F.conv2d(deltas, adv_filter, stride=1).squeeze()
        return advantages

    def _compute_loss(self, itr, obs, avail_actions, actions, rewards, valids, 
                      baselines, adjs, alive_masks):
        del itr

        if self.policy.recurrent:
            policy_entropies = self._compute_policy_entropy(obs, avail_actions, 
                actions=actions, adjs=adjs, alive_masks=alive_masks)
        else:
            policy_entropies = self._compute_policy_entropy(obs, avail_actions, 
                adjs=adjs, alive_masks=alive_masks)

        if self._maximum_entropy:
            rewards += self._policy_ent_coeff * policy_entropies

        max_path_length = min(self.max_path_length, max(valids).item()) # max actual length
        advantages = self.compute_advantages(max_path_length, baselines, rewards)

        if self._center_adv:
            means, variances = list(
                zip(*[(valid_adv.mean(), valid_adv.var(unbiased=False))
                      for valid_adv in filter_valids(advantages, valids)]))
            advantages = F.batch_norm(advantages.t(),
                                      torch.Tensor(means).to(self.device),
                                      torch.Tensor(variances).to(self.device),
                                      eps=self._eps).t()

        if self._positive_adv:
            advantages -= advantages.min()

        objective = self._compute_ppo_objective(advantages, obs, avail_actions, 
                                                actions, rewards, adjs, alive_masks)

        if self._entropy_regularzied:
            objective += self._policy_ent_coeff * policy_entropies

        valid_objectives = filter_valids(objective, valids)
        return -torch.cat(valid_objectives).mean()

    def _compute_policy_entropy(self, obs, avail_actions, actions=None, 
                                adjs=None, alive_masks=None):
        if self._stop_entropy_gradient:
            with torch.no_grad():
                if self.policy.recurrent:
                    policy_entropy = self.policy.entropy(obs, avail_actions, 
                        actions=actions, adjs=adjs, alive_masks=alive_masks)
                else:
                    policy_entropy = self.policy.entropy(obs, avail_actions, 
                        adjs=adjs, alive_masks=alive_masks)
        else:
            if self.policy.recurrent:
                policy_entropy = self.policy.entropy(obs, avail_actions, 
                    actions=actions, adjs=adjs, alive_masks=alive_masks)
            else:
                policy_entropy = self.policy.entropy(obs, avail_actions, 
                    adjs=adjs, alive_masks=alive_masks)

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_ppo_objective(self, advantages, obs, avail_actions, 
                               actions, rewards, adjs, alive_masks):
        # Compute constraint
        with torch.no_grad():
            old_ll = self._old_policy.log_likelihood(obs, avail_actions, 
                                                     actions, adjs, alive_masks)
        new_ll = self.policy.log_likelihood(obs, avail_actions, actions, 
                                            adjs, alive_masks)

        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        # Clipping the constraint
        likelihood_ratio_clip = torch.clamp(likelihood_ratio,
                                            min=1 - self._lr_clip_range,
                                            max=1 + self._lr_clip_range)

        # Calculate surrotate clip
        surrogate_clip = likelihood_ratio_clip * advantages

        return torch.min(surrogate, surrogate_clip)

    def process_paths(self, paths):
        for path in paths:
            if 'returns' not in path:
                path['returns'] = tensor_utils.discount_cumsum(
                    path['rewards'], self.discount)
        
    
        valids = torch.Tensor([len(path['actions']) for path in paths]).int()
        max_path_length = min(self.max_path_length, max(valids).item()) # max actual length
    
        returns = torch.stack([
            pad_to_last(tensor_utils.discount_cumsum(path['rewards'], self.discount).copy(),
                        total_length=max_path_length) for path in paths
        ])#.to(self.device)
        
        obses = torch.stack([
            pad_to_last(path['observations'],
                        total_length=max_path_length,
                        axis=0) for path in paths
        ])#.to(self.device) # (n_episodes, max_path_len, cent_obs_dim)
    
        states = torch.stack([
            pad_to_last(path['states'],
                        total_length=max_path_length,
                        axis=0) for path in paths
        ])#.to(self.device)
    
        actions = torch.stack([
            pad_to_last(path['actions'],
                        total_length=max_path_length,
                        axis=0) for path in paths
        ])#.to(self.device)

        avail_actions = torch.stack([
            pad_one_to_last(path['avail_actions'],
                        total_length=max_path_length,
                        axis=0) for path in paths
        ]) # Cannot pad all zero since prob sum cannot be zero
    
        rewards = torch.stack([
            pad_to_last(path['rewards'], total_length=max_path_length)
            for path in paths
        ])#.to(self.device)

        if self.policy.proximity_adj:
            adjs = torch.stack([
                pad_to_last(path['adjs'], 
                            total_length=max_path_length,
                            axis=0) for path in paths
            ])#.to(self.device) # (n_episodes, max_path_len, flat_adj_dim)
            alive_masks = None
        else:
            adjs = None
            alive_masks = torch.stack([
                pad_to_last(path['alive_masks'], 
                            total_length=max_path_length,
                            axis=0) for path in paths
            ])#.to(self.device) # (n_episodes, max_path_len, n_agents)

        # with torch.no_grad():
        #     baselines = self.baseline.forward(obses)

        return obses, avail_actions, actions, rewards, valids, \
            returns, adjs, alive_masks