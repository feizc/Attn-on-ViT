python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py  --batch_size 128 \
    --accum_iter 8 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
