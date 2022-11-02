# Monkey Island Stable Diffusion

[![Replicate](https://replicate.com/andreasjansson/monkey-island-sd/badge)](https://replicate.com/andreasjansson/monkey-island-sd)

Stable Diffusion 1.5 fine-tuned on frames from Monkey Island 1 and 2.

## Dataset

Frames were extracted from https://www.youtube.com/watch?v=QgRIXntFhww and https://www.youtube.com/watch?v=RkpG4FoszBQ.

Each frame was then captioned with https://replicate.com/salesforce/blip and the prefix `"Caption: "` was stripped out.

## Fine-tuning

First, run `cog run script/download-weights` to download the SD 1.5 weights.

The model was fine-tuned on a [customized version](https://github.com/andreasjansson/monkey-island-sd/blob/master/train_dreambooth.py) of the [Dreambooth training script in Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py).

Training parameters:
```
python train_dreambooth.py \
    --pretrained_model_name_or_path=diffusers-cache/models--runwayml--stable-diffusion-v1-5/snapshots/51b78538d58bd5526f1cf8e7c03c36e2799e0178 \
    --instance_data_dir=dataset \
    --output_dir=dreambooth-output \
    --resolution=144 \
    --train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1600 \
    --use_8bit_adam \
    --mixed_precision=fp16 \
    --class_data_dir=class-data-dir \
    --with_prior_preservation \
    --num_class_images=2000 \
    --train_text_encoder
```

There's still room for improvement. It doesn't always get the prompt right, especially if there are multiple concepts in the prompt. There's some degree of catastrophic forgetting going on.
