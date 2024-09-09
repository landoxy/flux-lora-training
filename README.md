# FLUX LORA TRAINING
Lora fine-tuning process for the latest flux-dev model. Might not be perfect (flux just came out).

Model link: [Huggingface FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

## General workflow
1. Get your hands on a GPU with 80GB VRAM (I use a A100 on Azure)
2. Clone diffusers and install python packages: 
```shell
pip install torch torchvision 
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
cd diffusers/examples/
pip install -r requirements_flux.txt
```
3. There you find a lora training script for flux `diffusers/examples/dreambooth/train_dreambooth_lora_flux.py.py`
4. You can pretty much follow `diffusers/examples/dreambooth/README_flux.md`


5. My training script `train-custom-lora.sh` that I execute in `diffusers/examples/dreambooth/`
```shell
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="face_pictures"
export OUTPUT_DIR="trained-flux-lora-face-pictures"

accelerate launch train_dreambooth_lora_flux.py --pretrained_model_name_or_path=$MODEL_NAME  --instance_data_dir=$INSTANCE_DIR  --instance_prompt="a photo of a xyz man" --output_dir=$OUTPUT_DIR --mixed_precision="bf16" --rank 128 --resolution=1024 --optimizer="prodigy" --train_batch_size=1 --gradient_accumulation_steps=2 --guidance_scale=1  --learning_rate=1. --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=1500  --seed="0" --checkpointing_steps=100 --dataloader_num_workers=2 --center_crop
```
Put your pictures in `diffusers/examples/dreambooth/[INSTANCE_DIR]/` and adjust `--instance_prompt` to match the subject you are fine-tuning. For me it was face pictures.

## Insights
- Rank 128 works best for me with around 1500 training steps
- `--optimizer="prodigy"` with `--learning_rate=1.` makes life much easier, less hyperparameter to tune (it just works!)
- Sometimes lora can lower the overall quality of the model, adjusting the adapter_weight or mixing it with other loras to balance can help



## FAQ
There might be some errors since flux is new and the example scripts provided in diffusers are not perfect.

- Error message similar to `RuntimeError: Input type (float) and bias type (c10::Half) should be the same`

    [https://github.com/huggingface/diffusers/issues/9237#issuecomment-2309643184](https://github.com/huggingface/diffusers/issues/9237#issuecomment-2309643184) fixed it for me

- Torch warning `torch tensor 3d is deprecated use 2d` (or similar)

    [https://github.com/huggingface/diffusers/issues/9350](https://github.com/huggingface/diffusers/issues/9350) addresses it
