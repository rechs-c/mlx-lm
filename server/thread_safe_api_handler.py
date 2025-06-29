import threading
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, Sequence

from mlx_lm.server import (
    APIHandler,
    PromptCache,
    get_system_fingerprint,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
    common_prefix_len,
)

class ThreadSafeAPIHandler(APIHandler):
    def __init__(
        self,
        model_provider, # 这里不再是 ModelProvider，而是 ThreadSafeModelProvider
        *args,
        prompt_cache: Optional[PromptCache] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_provider,
            *args,
            prompt_cache=prompt_cache,
            system_fingerprint=system_fingerprint,
            **kwargs,
        )
        self.prompt_cache_lock = threading.Lock() # 新增：用于保护 prompt_cache 的锁

    def reset_prompt_cache(self, prompt):
        with self.prompt_cache_lock: # 使用锁保护缓存重置操作
            logging.debug(f"*** Resetting cache. ***")
            self.prompt_cache.model_key = self.model_provider.model_key
            self.prompt_cache.cache = make_prompt_cache(self.model_provider.model)
            if self.model_provider.draft_model is not None:
                self.prompt_cache.cache += make_prompt_cache(
                    self.model_provider.draft_model
                )
            self.prompt_cache.tokens = list(prompt)  # Cache the new prompt fully

    def get_prompt_cache(self, prompt):
        with self.prompt_cache_lock: # 使用锁保护缓存获取和修改操作
            cache_len = len(self.prompt_cache.tokens)
            prompt_len = len(prompt)
            com_prefix_len = common_prefix_len(self.prompt_cache.tokens, prompt)

            # Leave at least one token in the prompt
            com_prefix_len = min(com_prefix_len, len(prompt) - 1)

            # Condition 1: Model changed or no common prefix at all. Reset cache.
            if (
                self.prompt_cache.model_key != self.model_provider.model_key
                or com_prefix_len == 0
            ):
                self.reset_prompt_cache(prompt)

            # Condition 2: Common prefix exists and matches cache length. Process suffix.
            elif com_prefix_len == cache_len:
                logging.debug(
                    f"*** Cache is prefix of prompt (cache_len: {cache_len}, prompt_len: {prompt_len}). Processing suffix. ***"
                )
                prompt = prompt[com_prefix_len:]
                self.prompt_cache.tokens.extend(prompt)

            # Condition 3: Common prefix exists but is shorter than cache length. Attempt trim.
            elif com_prefix_len < cache_len:
                logging.debug(
                    f"*** Common prefix ({com_prefix_len}) shorter than cache ({cache_len}). Attempting trim. ***"
                )

                if can_trim_prompt_cache(self.prompt_cache.cache):
                    num_to_trim = cache_len - com_prefix_len
                    logging.debug(f"    Trimming {num_to_trim} tokens from cache.")
                    trim_prompt_cache(self.prompt_cache.cache, num_to_trim)
                    self.prompt_cache.tokens = self.prompt_cache.tokens[:com_prefix_len]
                    prompt = prompt[com_prefix_len:]
                    self.prompt_cache.tokens.extend(prompt)
                else:
                    logging.debug(f"    Cache cannot be trimmed. Resetting cache.")
                    self.reset_prompt_cache(prompt)

            # This case should logically not be reached if com_prefix_len <= cache_len
            else:
                logging.error(
                    f"Unexpected cache state: com_prefix_len ({com_prefix_len}) > cache_len ({cache_len}). Resetting cache."
                )
                self.reset_prompt_cache(prompt)

            logging.debug(f"Returning {len(prompt)} tokens for processing.")
            return prompt
