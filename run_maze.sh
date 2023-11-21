python ./train.py \
--env PointMaze_Large-v3 \
--algo tqc \
--wandb-project-name baselines \
--track \
--save-freq 1000 \
--save-replay-buffer \
--env-kwargs continuing_task:False \
-tags maze,no_continuing_task \
-params n_timesteps:1000000
