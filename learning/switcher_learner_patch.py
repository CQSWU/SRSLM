"""Sample Factory loss guards plus SRSLM Switcher actor masking.

The Switcher does not make a policy decision when AORePlan returns wait.  Such
transitions still train the critic, but they must not enter PPO advantage
normalization or any actor loss.  Sample Factory has only one validity mask,
so this module narrows that mask while computing the actor losses and then
recomputes the value loss with the original mask.

Sample Factory 2.1.1 also normalizes advantages with the default unbiased
``torch.std_mean`` estimator.  One valid row therefore yields NaN, while zero
valid rows make every masked mean undefined.  The project-level wrapper keeps
the upstream implementation byte-for-byte for two or more valid rows.  Only
the degenerate boundary is handled specially: one row uses the population
standard deviation (necessarily zero), and zero rows are rejected before a
forward or optimizer step.  This guard applies to every project policy, including the v4
trace actors, and does not alter ordinary PPO minibatches.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

import torch


_PATCH_ATTRIBUTE = "_srslm_switcher_actor_mask_patch"
_VALID_COUNT_ATTRIBUTE = "_srslm_valid_count_guard"
_STD_MEAN_PATCH_LOCK = RLock()


class _DegenerateStdMeanTorchProxy:
    """Forward every torch API except the default std_mean boundary case."""

    def __init__(self, base: Any):
        self._base = base

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def std_mean(self, inputs, *args, **kwargs):
        # Intercept only Sample Factory's default unbiased call.  Explicit
        # correction/unbiased requests elsewhere retain their exact meaning.
        if not args and not kwargs and inputs.numel() <= 1:
            if inputs.numel() == 1:
                return self._base.std_mean(inputs, correction=0)
            zero = inputs.new_zeros(())
            return zero, zero
        return self._base.std_mean(inputs, *args, **kwargs)


def _invalid_count(valids: torch.Tensor) -> int:
    return int(valids.numel() - valids.sum().item())


def _switch_allowed_mask(mb: Any) -> torch.Tensor:
    actions = mb.normalized_obs["aoreplan_action"]
    if actions.ndim != 2 or actions.shape[-1] != 5:
        raise ValueError(
            "Switcher learner expects flattened five-way AORePlan one-hot "
            f"actions, got shape {tuple(actions.shape)}."
        )
    return torch.argmax(actions, dim=-1).ne(0)


def _call_with_degenerate_std_guard(
    original: Callable,
    learner: Any,
    mb: Any,
    num_invalids: int,
):
    """Run the unmodified upstream loss with a module-local torch proxy.

    A Python function resolves ``torch`` through its own globals dictionary.
    Replacing that one binding avoids modifying the process-wide torch module.
    The lock makes restoration deterministic even if a test calls the learner
    from more than one thread.
    """

    function_globals = original.__globals__
    with _STD_MEAN_PATCH_LOCK:
        base_torch = function_globals.get("torch")
        if base_torch is None or not hasattr(base_torch, "std_mean"):
            raise RuntimeError(
                "Cannot guard Sample Factory advantage normalization: the "
                "upstream loss function has no torch.std_mean binding."
            )
        function_globals["torch"] = _DegenerateStdMeanTorchProxy(base_torch)
        try:
            return original(learner, mb, num_invalids)
        finally:
            function_globals["torch"] = base_torch


def _calculate_with_valid_count_guard(
    original: Callable,
    learner: Any,
    mb: Any,
    num_invalids: int,
):
    """Preserve upstream for >=2 rows; define the exact 1/0-row boundary."""

    valid_count = int(mb.valids.bool().sum().item())
    if valid_count >= 2:
        return original(learner, mb, num_invalids)
    if valid_count == 0:
        raise RuntimeError(
            "Sample Factory produced a minibatch with zero valid samples; "
            "the update is rejected before loss evaluation to avoid NaN and "
            "optimizer-state drift."
        )

    degenerate_num_invalids = _invalid_count(mb.valids.bool())
    return _call_with_degenerate_std_guard(
        original, learner, mb, degenerate_num_invalids
    )


def _build_switcher_calculate_losses(original: Callable) -> Callable:
    def patched(self, mb, num_invalids):
        if getattr(self.cfg, "encoder_custom", None) != "switcher":
            return _calculate_with_valid_count_guard(
                original, self, mb, num_invalids
            )

        if self.cfg.use_rnn:
            raise RuntimeError("Switcher actor masking requires use_rnn=false.")
        if self.cfg.with_vtrace:
            raise RuntimeError("Switcher actor masking requires with_vtrace=false.")
        if self.cfg.normalize_input:
            raise RuntimeError(
                "Switcher actor masking requires normalize_input=false."
            )

        original_valids = mb.valids
        full_valids = original_valids.bool()
        switch_allowed = _switch_allowed_mask(mb)
        if switch_allowed.shape != full_valids.shape:
            raise ValueError(
                "AORePlan action mask and Sample Factory validity mask have "
                f"different shapes: {tuple(switch_allowed.shape)} versus "
                f"{tuple(full_valids.shape)}."
            )

        actor_valids = full_valids & switch_allowed
        full_num_invalids = _invalid_count(full_valids)
        full_sample_count = int(full_valids.sum().item())
        actor_num_invalids = _invalid_count(actor_valids)
        actor_sample_count = int(actor_valids.sum().item())

        # torch.std_mean uses the unbiased estimator in Sample Factory 2.1.1,
        # so zero or one selected actor sample would produce NaN.  In that
        # degenerate case we perform the full forward pass and deliberately
        # make the actor update zero; the critic still uses every valid row.
        if actor_sample_count >= 2:
            mb.valids = actor_valids
            try:
                losses = _calculate_with_valid_count_guard(
                    original, self, mb, actor_num_invalids
                )
            finally:
                mb.valids = original_valids
        else:
            losses = _calculate_with_valid_count_guard(
                original, self, mb, full_num_invalids
            )

        (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            _masked_value_loss,
            loss_summaries,
        ) = losses

        if full_sample_count:
            values = loss_summaries["values"].squeeze(-1)
            value_loss = self._value_loss(
                values,
                mb["values"],
                mb.returns,
                self.cfg.ppo_clip_value,
                full_valids,
                full_num_invalids,
            )

        if actor_sample_count < 2:
            # Keep the zero connected to the model graph so loss.backward()
            # remains valid even when the minibatch contains only waits.
            zero = loss_summaries["values"].sum() * 0.0
            policy_loss = zero
            exploration_loss = zero
            kl_loss = zero
            # _train() always takes mean/max of kl_old for LR scheduling.
            kl_old = zero.detach().reshape(1)
            loss_summaries = dict(loss_summaries)
            loss_summaries["adv"] = torch.zeros_like(loss_summaries["adv"])
            loss_summaries["adv_mean"] = zero.detach()
            loss_summaries["adv_std"] = zero.detach()

        return (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
        )

    setattr(patched, _PATCH_ATTRIBUTE, True)
    setattr(patched, _VALID_COUNT_ATTRIBUTE, True)
    patched._srslm_original_calculate_losses = original
    return patched


def patch_switcher_learner_losses() -> None:
    """Install the valid-count guard and Switcher mask once per process."""

    from sample_factory.algo.learning.learner import Learner

    current = Learner._calculate_losses
    if getattr(current, _PATCH_ATTRIBUTE, False):
        return
    Learner._calculate_losses = _build_switcher_calculate_losses(current)
