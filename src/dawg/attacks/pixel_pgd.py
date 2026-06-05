"""Pixel-space PGD (L1 untargeted) against MolmoWeb-4B-Native.

The honest attack: optimize the actual screenshot pixels inside a bbox, with the
differentiable preprocessing bridge carrying gradients pixel -> images tensor ->
vision tower -> LM -> CE loss. L1 untargeted = MAXIMIZE cross-entropy against
MolmoWeb's clean answer (push its answer away from correct).

Gradient chain:
    delta (bbox pixels)             # leaf, requires_grad
      -> adv = clamp(screenshot + delta*mask, 0, 255)        # [0,255]
      -> images = bridge(adv)                                # [0,255] float
      -> images / 255.0                                      # [0,1] float
      -> model.forward(images=...)   # normalize_image_tensor float-branch -> [-1,1]
      -> CE(target = clean-answer tokens)

This module sets everything up and exposes `sanity_check` (one forward+backward,
no optimization) for the "set it up but don't run the attack yet" milestone, plus
`pgd_l1` for later. Self-contained (does not import the old patch-token-space attack stack).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dawg.attacks.diff_preprocess import DiffPreprocessor, find_multicrop_preprocessor


# --------------------------------------------------------------------------- #
# Batch building (clean-answer tokens appended as the CE target)
# --------------------------------------------------------------------------- #

def build_batch_with_target(pred, img_pil: Image.Image, question: str,
                            target_text: str, device) -> tuple[dict, int, int]:
    """Preprocess (prompt + image) then append tokenized `target_text` to the
    per-token fields so model.forward yields a CE loss on the answer.

    Appends the tokenized answer as the CE target (no add_special_tokens kwarg —
    HfTokenizerWrapper.encode doesn't accept it)."""
    batch = pred.preprocessor(dict(image=img_pil, style="demo", question=question))
    batch["input_ids"] = batch.pop("input_tokens")
    batch.pop("metadata", None)

    tokenizer = pred.preprocessor.preprocessor.text_preprocessor.tokenizer
    target_ids = np.array(tokenizer.encode(target_text), dtype=batch["input_ids"].dtype)
    n_t = len(target_ids)

    batch["input_ids"] = np.concatenate([batch["input_ids"], target_ids])
    batch["target_tokens"] = np.concatenate([batch["target_tokens"], target_ids])
    batch["loss_masks"] = np.concatenate(
        [batch["loss_masks"], np.ones(n_t, dtype=batch["loss_masks"].dtype)])
    last_pos = int(batch["position_ids"][-1])
    batch["position_ids"] = np.concatenate(
        [batch["position_ids"],
         np.arange(last_pos + 1, last_pos + 1 + n_t, dtype=batch["position_ids"].dtype)])

    base_n = len(batch["input_ids"]) - n_t
    batch_t = {k: torch.as_tensor(np.expand_dims(v, 0), device=device)
               for k, v in batch.items() if isinstance(v, np.ndarray)}
    return batch_t, base_n, n_t


def _call_model_forward(model, batch_t: dict):
    # Molmo.forward accepts ONLY these image/LM kwargs (verified signature). It
    # returns OLMoOutput(logits=...) and computes NO loss internally — no
    # target_tokens/loss_masks/cum_* (those were run.py's unverified guess).
    candidate = {k: batch_t.get(k) for k in (
        "input_ids", "images", "image_masks", "token_pooling", "position_ids")}
    return model(**{k: v for k, v in candidate.items() if v is not None})


def _compute_lm_loss(output, target_tokens, loss_masks):
    if getattr(output, "loss", None) is not None and output.loss.requires_grad:
        return output.loss
    logits = output.logits if hasattr(output, "logits") else output
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, target_tokens.unsqueeze(-1).long()).squeeze(-1)
    mask = loss_masks.float()
    return (nll * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def bbox_mask(H: int, W: int, bboxes, device) -> torch.Tensor:
    """(H,W,3) float mask, 1 inside the box(es). `bboxes` is a single [x,y,w,h]
    or a list of them (union mask -> multi-patch attacks)."""
    boxes = bboxes if (len(bboxes) > 0 and isinstance(bboxes[0], (list, tuple))) else [bboxes]
    m = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    for (x, y, w, h) in boxes:
        m[int(y):int(y + h), int(x):int(x + w), :] = 1.0
    return m


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

class PixelPGDSetup:
    """Everything needed to run / sanity-check pixel-space PGD on one example."""

    def __init__(self, pred, image_np: np.ndarray, question: str,
                 bbox: tuple[int, int, int, int], *, device=None, clean_answer=None):
        self.pred = pred
        self.device = device or next(pred.model.parameters()).device
        self.model_dtype = next(pred.model.parameters()).dtype
        # We only need gradients w.r.t. the pixel delta, not the weights —
        # freezing params keeps backward from allocating param-grad buffers.
        for p in pred.model.parameters():
            p.requires_grad_(False)
        self.question = question
        # bbox is a single [x,y,w,h] or a list of them (multi-patch union mask)
        if len(bbox) > 0 and isinstance(bbox[0], (list, tuple)):
            self.bboxes = [[int(v) for v in b] for b in bbox]
        else:
            self.bboxes = [[int(v) for v in bbox]]
        self.bbox = tuple(self.bboxes[0])  # first box (compat / sanity stat)

        self.image_np = image_np
        self.img_pil = Image.fromarray(image_np.astype("uint8")).convert("RGB")
        self.H, self.W = image_np.shape[:2]

        # Clean answer = the CE target the L1 attack pushes away from.
        self.clean_answer = clean_answer if clean_answer is not None else \
            pred.predict(question, image_np)

        # Non-pixel batch fields (geometry/tokens) from the real preprocessor.
        self.batch_t, self.n_prompt, self.n_target = build_batch_with_target(
            pred, self.img_pil, question, self.clean_answer, self.device)
        self.ref_images = self.batch_t["images"]  # real uint8->tensor, for reference

        # Differentiable bridge configured from the live preprocessor.
        self.params = find_multicrop_preprocessor(pred.preprocessor)
        self.bridge = DiffPreprocessor(**self.params)

        self.screenshot = torch.tensor(image_np.astype("float32"), device=self.device)  # (H,W,3)[0,255]
        self.mask = bbox_mask(self.H, self.W, self.bboxes, self.device)
        self.n_perturbable = int(self.mask.sum().item())

    def adv_images(self, delta: torch.Tensor) -> torch.Tensor:
        """delta (H,W,3) -> images (1, n_crops, n_patches, 588) in [0,1], model dtype."""
        adv = (self.screenshot + delta * self.mask).clamp(0.0, 255.0)
        images = self.bridge(adv)                      # (n_crops,729,588) [0,255]
        images = (images / 255.0).unsqueeze(0)         # (1,...) [0,1]
        return images.to(self.device, self.model_dtype)

    def loss_at(self, delta: torch.Tensor) -> torch.Tensor:
        """Teacher-forced CE on the clean-answer span (L1 untargeted MAXIMIZES this).

        Molmo.forward returns logits only, so we compute the next-token CE over
        the appended answer tokens directly from (n_prompt, n_target): logits at
        positions [n_prompt-1, n_prompt+n_target-1) predict input_ids at
        [n_prompt, n_prompt+n_target). This avoids any ambiguity about how the
        preprocessor's target_tokens/loss_masks are shifted."""
        self.batch_t["images"] = self.adv_images(delta)
        out = _call_model_forward(self.pred.model, self.batch_t)
        logits = out.logits if hasattr(out, "logits") else out
        lo, hi = self.n_prompt - 1, self.n_prompt + self.n_target - 1
        sel = logits[0, lo:hi, :].float()
        labels = self.batch_t["input_ids"][0, self.n_prompt:self.n_prompt + self.n_target].long()
        return F.cross_entropy(sel, labels)

    # ---------------- sanity check (one forward + backward, NO optimization) ---
    def sanity_check(self) -> dict:
        self.pred.model.eval()
        delta = torch.zeros_like(self.screenshot, requires_grad=True)
        loss = self.loss_at(delta)
        loss.backward()
        g = delta.grad
        x, y, w, h = self.bbox
        in_box = g[y:y + h, x:x + w, :]
        out_box_sum = g.abs().sum().item() - in_box.abs().sum().item()
        # bridge fidelity at delta=0 vs the real images (both in [0,255])
        with torch.no_grad():
            mine = self.bridge(self.screenshot)
            bridge_max_diff = (mine - self.ref_images.float().squeeze(0)).abs().max().item()
        return {
            "clean_answer": self.clean_answer,
            "n_prompt_tokens": self.n_prompt,
            "n_target_tokens": self.n_target,
            "n_crops": int(self.ref_images.shape[1]),
            "images_shape": tuple(self.ref_images.shape),
            "n_perturbable_pixels": self.n_perturbable,
            "loss": float(loss.detach().item()),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "grad_max_abs": float(g.abs().max().item()),
            "grad_mean_abs_in_box": float(in_box.abs().mean().item()),
            "grad_abs_sum_outside_box": float(out_box_sum),
            "bridge_max_diff_0to255": float(bridge_max_diff),
        }

    # ---------------- full PGD (for LATER; not used by the sanity milestone) ---
    def pgd_l1(self, *, eps: float = 16.0, n_iter: int = 50, lr: float = 2.0,
               verbose: bool = True, delta0: torch.Tensor | None = None,
               optim: str = "sign"):
        """optim: 'sign' = Linf sign-PGD (default, unchanged); 'momentum' = MI-FGSM
        (accumulate normalized grad, step on its sign); 'adam' = Adam in pixel space,
        projected back to the eps-ball + patch mask each step.
        delta0: optional starting perturbation (for random restarts)."""
        self.pred.model.eval()
        if delta0 is None:
            delta = torch.zeros_like(self.screenshot, requires_grad=True)
        else:
            delta = (delta0.to(self.screenshot) * self.mask).clamp(-eps, eps) \
                .detach().clone().requires_grad_(True)
        history = []
        g_mom = torch.zeros_like(delta)
        opt = torch.optim.Adam([delta], lr=lr, maximize=True) if optim == "adam" else None
        for step in range(n_iter):
            loss = self.loss_at(delta)
            if optim == "adam":
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    delta.mul_(self.mask).clamp_(-eps, eps)  # project to mask + eps-ball
            else:
                if delta.grad is not None:
                    delta.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    g = delta.grad * self.mask
                    if optim == "momentum":
                        g_mom.add_(g / (g.abs().mean() + 1e-12))  # MI-FGSM, mu=1
                        step_dir = g_mom.sign()
                    else:  # 'sign'
                        step_dir = g.sign()
                    delta.add_(lr * step_dir * self.mask).clamp_(-eps, eps)
            history.append(float(loss.detach().item()))
            if verbose:
                print(f"  pgd[{optim}] {step:3d}: loss={history[-1]:.4f} "
                      f"|delta|max={delta.detach().abs().max().item():.1f}")
        with torch.no_grad():
            adv = (self.screenshot + delta * self.mask).clamp(0, 255)
        return adv.detach().cpu().numpy().astype("uint8"), history
