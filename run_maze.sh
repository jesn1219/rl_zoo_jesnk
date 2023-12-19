python ./train.py \
--env PointMaze_UMaze-v3 \
--algo tqc \
--wandb-project-name baselines \
#--track \
--save-freq 10000 \
--save-replay-buffer \
--env-kwargs continuing_task:False \
-tags maze,no_continuing_task,g098 \
-params n_timesteps:1000000 gamma:0.98
