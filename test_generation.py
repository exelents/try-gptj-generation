
import torch
import gptj_wrapper as gptj
import datetime

model = gptj.GPTJ(stage=3)

eos_newline = model.tokenizer("<|endoftext|>")['input_ids'][0]

with torch.no_grad():
    text = """I am a highly intelligent question answering bot. If you provide me a context and ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer based on the context, I will respond with "Unknown".

Context: In 2017, U.S. life expectancy was 78.6 years.
Question: What is human life expectancy in the United States?
Answer: 78 years.

Context: puppy A is happy. puppy B is sad.
Question: which puppy is happy?
Answer: puppy A.

Context: You poured a glass of cranberry, but then absentmindedly, you poured about a teaspoon of grape juice into it. It looks OK. You try sniffing it, but you have a bad cold, so you can't smell anything. You are very thirsty. So you drink it.
Question: What happens next?
Answer:
"""
    start = datetime.datetime.now()
    out = model.generate(
        text=text,
        max_length=512,
#         num_beams=5,
        do_sample=True,
        temperature=0.1,
        top_k=5,
        top_p=0.9,
        no_repeat_ngram_size=2, 
        early_stopping=True,
#         num_return_sequences=1,
        use_cache=False,
        eos_token_id=eos_newline
    )
    duration = datetime.datetime.now() - start
    for o in out:
        print("\n\n\n")
        print(o[len(text):])
    print(f"\n\nDuration = {duration.total_seconds()}")
