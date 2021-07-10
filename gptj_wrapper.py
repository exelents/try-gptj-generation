import os
import torch
import json
from transformers import GPTNeoForCausalLM, AutoConfig, AutoTokenizer
import deepspeed
import torch
import argparse
from utils import get_argument_parser
from transformers import GPTNeoPreTrainedModel
from typing import Union, Iterable, Tuple


NoneType = type(None)

dist_env_1_gpu = dict(MASTER_ADDR="localhost", MASTER_PORT="10999", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
for k,v in dist_env_1_gpu.items():
    os.environ[k] = v


def get_model_config_tokenizer(model_path):
    # GPT-J 6B config
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
    config.attention_layers = ["global"] * 28
    config.attention_types = [["global"], 28]
    config.num_layers = 28
    config.num_heads = 16
    config.hidden_size = 256 * config.num_heads
    config.vocab_size = 50400
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True

    try:
        from collections.abc import MutableMapping
    except ImportError:
        from collections import MutableMapping
    from pathlib import Path

    class Checkpoint(MutableMapping):
        def __init__(self, chkpt_dir, device="cpu"):
            self.device = device
            self.chkpt_dir = Path(chkpt_dir)
            self.checkpoint = torch.load(str(chkpt_dir / Path("m.pt")))
        def __len__(self):
            return len(self.checkpoint)
        def __getitem__(self, key):
            path = self.chkpt_dir / Path(self.checkpoint[key]).name
            return torch.load(str(path), map_location=self.device)
        def __setitem__(self, key, value):
            return
        def __delitem__(self, key, value):
            return
        def keys(self):
            return self.checkpoint.keys()
        def __iter__(self):
            for key in self.checkpoint:
                yield (key, self.__getitem__(key))
        def __copy__(self):
            return Checkpoint(self.chkpt_dir, device=self.device)
        def copy(self):
            return Checkpoint(self.chkpt_dir, device=self.device)

    model = GPTNeoForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None,
        config=config,
        state_dict=Checkpoint(model_path)
    )

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    return model, config, tokenizer


def get_deepspeed_engine_optimizer(model, stage=2):
    if stage == 3:
        config_filename = 'ds_config_stage3.json'
    elif stage == 2:
        config_filename = 'ds_config_stage2.json'
    else:
        raise Exception('Wrong stage number')
    deepspeed.init_distributed(dist_backend='nccl')

    parser = get_argument_parser()
    parser = deepspeed.add_config_arguments(parser)
    args_unparsed = f"--train_batch_size 16 --deepspeed --deepspeed_config {config_filename} --output_dir ./output_dir".split()
    args = parser.parse_args(args_unparsed)
    args.local_rank = int(os.environ['LOCAL_RANK']) if args.local_rank != -1 else args.local_rank

    config_params = json.load(open(args.deepspeed_config))
    config_params['train_batch_size'] = args.train_batch_size

    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters(),
                                                         config_params=config_params)
    return model_engine, optimizer


class GPTJ(GPTNeoPreTrainedModel):
    def __init__(self, model_path="j6b_ckpt", seq_len=512, stage=0):
        assert stage in [0, 2, 3], 'Support only stage levels 2/3 or no deepspeed (stage==0)'
        optimizer = None
        model, config, tokenizer = get_model_config_tokenizer(model_path)
        if stage in [2, 3]:
            model, optimizer = get_deepspeed_engine_optimizer(model, stage=stage)

        super().__init__(config)

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.seq_len = seq_len
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.stage = stage
        if stage == 0:
            self.model.to('cuda')

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PretrainedModel.beam_search` or :meth:`~transformers.PretrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        kwargs.update({"input_ids": input_ids})
        return kwargs

    def generate(
            self, text: Union[str, NoneType] = None,
            input_ids: Union[torch.LongTensor, NoneType] = None,
            max_length: Union[int, None] = None,
            min_length: Union[int, NoneType] = None,
            do_sample: Union[bool, NoneType] = None,
            early_stopping: Union[bool, NoneType] = None,
            num_beams: Union[int, NoneType] = None,
            temperature: Union[float, NoneType] = None,
            top_k: Union[int, NoneType] = None,
            top_p: Union[float, NoneType] = None,
            repetition_penalty: Union[float, NoneType] = None,
            bad_words_ids: Union[Iterable[int], NoneType] = None,
            bos_token_id: Union[int, NoneType] = None,
            pad_token_id: Union[int, NoneType] = None,
            eos_token_id: Union[int, NoneType] = None,
            length_penalty: Union[float, NoneType] = None,
            no_repeat_ngram_size: Union[int, NoneType] = None,
            num_return_sequences: Union[int, NoneType] = None,
            decoder_start_token_id: Union[int, NoneType] = None,
            use_cache: Union[bool, NoneType] = None,
            **model_kwargs):
        if text is not None:
            input_ids = torch.cuda.LongTensor([self.tokenizer(text)['input_ids']])
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if self.stage == 0:
            input_ids.to('cuda')
        res = super().generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            **model_kwargs
        )
        if self.stage == 0:
            res.detach().to('cpu')
        return list(map(self.tokenizer.decode, res.tolist()))

    def __call__(self, *args, **kwargs):
        if 'past' in kwargs:
            kwargs.pop('past')
        return self.model(**kwargs)
