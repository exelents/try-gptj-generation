## Generation with GPT-J on DeepSpeed

Use deepspeed_wrapper.ipynb for generation.

The point of this wrapper on GPT-J model is application of Deepspeed library 
allows to reduce GPU memory usage. It works for me on RTX 3090 and DS stage 2.
Maybe on stage 3 it can run on GPUs with even less memory.

conversion scripts are taken from:
https://gist.github.com/finetuneanon/7dd417a31338a63f219a49702e0550db
https://gist.github.com/finetuneanon/b814e702b1e2e150d2e263f6a75f4650