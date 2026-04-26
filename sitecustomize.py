from __future__ import annotations


def _patch_transformers_tokenizer_compat() -> None:
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )


_patch_transformers_tokenizer_compat()
