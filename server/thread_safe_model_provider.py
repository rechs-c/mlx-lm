import argparse
import threading
import logging
from pathlib import Path
from typing import Optional, Tuple

from mlx_lm.server import ModelProvider, load # 导入原始的 ModelProvider 和 load 函数

class ThreadSafeModelProvider(ModelProvider):
    def __init__(self, cli_args: argparse.Namespace):
        super().__init__(cli_args)
        self.lock = threading.Lock() # 新增：线程锁

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        with self.lock: # 使用锁保护加载操作
            if self.model_key == (model_path, adapter_path, draft_model_path):
                return self.model, self.tokenizer

            # Remove the old model if it exists.
            self.model = None
            self.tokenizer = None
            self.model_key = None
            self.draft_model = None

            # Building tokenizer_config
            tokenizer_config = {
                "trust_remote_code": True if self.cli_args.trust_remote_code else None
            }
            if self.cli_args.chat_template:
                tokenizer_config["chat_template"] = self.cli_args.chat_template

            if model_path == "default_model":
                if self.cli_args.model is None:
                    raise ValueError(
                        "A model path has to be given as a CLI "
                        "argument or in the HTTP request"
                    )
                model, tokenizer = load(
                    self.cli_args.model,
                    adapter_path=(
                        adapter_path if adapter_path else self.cli_args.adapter_path
                    ),
                    tokenizer_config=tokenizer_config,
                )
            else:
                self._validate_model_path(model_path)
                model, tokenizer = load(
                    model_path, adapter_path=adapter_path, tokenizer_config=tokenizer_config
                )

            if self.cli_args.use_default_chat_template:
                if tokenizer.chat_template is None:
                    tokenizer.chat_template = tokenizer.default_chat_template

            self.model_key = (model_path, adapter_path, draft_model_path)
            self.model = model
            self.tokenizer = tokenizer

            def validate_draft_tokenizer(draft_tokenizer):
                if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                    logging.warning(
                        "Draft model tokenizer does not match model tokenizer. "
                        "Speculative decoding may not work as expected."
                    )

            if (
                draft_model_path == "default_model"
                and self.cli_args.draft_model is not None
            ):
                self.draft_model, draft_tokenizer = load(self.cli_args.draft_model)
                validate_draft_tokenizer(draft_tokenizer)

            elif draft_model_path is not None and draft_model_path != "default_model":
                self._validate_model_path(draft_model_path)
                self.draft_model, draft_tokenizer = load(draft_model_path)
                validate_draft_tokenizer(draft_tokenizer)
            return self.model, self.tokenizer
