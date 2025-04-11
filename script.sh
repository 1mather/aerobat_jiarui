

#train vqbet model
#select_fruit_none_distractor_v3_10
python3 lerobot/scripts/train.py --policy.type vqbet --dataset.repo_id lerobot/select_fruit_none_distractor_v3_100 --dataset.root ~/.cache/huggingface/lerobot/select_fruit_none_distractor_v3_100\
    --seed=100000 \
    --batch_size=20 \
    --steps=100000 \
    --eval_freq=10000 \
    --save_freq=10000 \
    --wandb.enable=true\
    --dataset.num_workers=16
#pushT
python lerobot/scripts/train.py \
    --output_dir=outputs/train/vqbet_pusht \
    --policy.type=vqbet \
    --dataset.repo_id=lerobot/pusht \
    --env.type=pusht \
    --seed=100000 \
    --batch_size=64 \
    --steps=250000 \
    --eval_freq=25000 \
    --save_freq=25000 \
    --wandb.enable=true

#Evaluate vqbet model

python lerobot/scripts/eval.py \
    --policy.path=outputs/train/vqbet_pusht/checkpoints/200000/pretrained_model \
    --output_dir=outputs/eval/vqbet_pusht/200000 \
    --env.type=pusht \
    --seed=100000 \
    --eval.n_episodes=500 \
    --eval.batch_size=50 \
    --device=cuda \
    --use_amp=false
