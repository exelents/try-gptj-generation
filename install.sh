apt install zstd

time wget -c https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd

time tar -I zstd -xf step_383500_slim.tar.zstd

git clone https://github.com/kingoflolz/mesh-transformer-jax.git
pip install -r mesh-transformer-jax/requirements.txt

pip install mesh-transformer-jax/ jax==0.2.12

git clone https://github.com/finetuneanon/transformers
git -C ./transformers checkout gpt-j
pip install transformers/

pip install deepspeed

python3 ./conv.py
