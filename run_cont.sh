#!/bin/bash

# Create a tmux session (session name 'gpumon')
echo "Creating tmux session 'gpumon'..."
tmux new-session -d -s gpumon

while true; do
    # Check the total and available GPU memory
    echo "Checking GPU memory..."
    gpu_ram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    gpu_ram_available=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    echo "Total GPU RAM: $gpu_ram MB, Available GPU RAM: $gpu_ram_available MB"

    # Generate a unique identifier for each run
    unique_id=$(date +%s)

    # Create a new window for running 'run.sh' and execute it
    echo "Running 'run.sh'..."
    window_name="run_script_$unique_id"
    tmux new-window -t gpumon -n $window_name
    tmux send-keys -t gpumon:$window_name './run.sh' C-m
    sleep 15

    # Check GPU memory usage after running 'run.sh'
    echo "Checking GPU memory usage after running 'run.sh'..."
    gpu_ram_after=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    run_gpu_occupied=$((gpu_ram_available - gpu_ram_after))
    echo "GPU RAM used by 'run.sh': $run_gpu_occupied MB"

    # Compare the current available GPU RAM with the usage after running 'run.sh'
    if (( gpu_ram_after > run_gpu_occupied + 600 )); then
        echo "Sufficient memory available, running 'run.sh' again..."
        unique_id=$(date +%s)
        window_name="run_script_$unique_id"
        tmux new-window -t gpumon -n $window_name
        tmux send-keys -t gpumon:$window_name './run.sh' C-m
    else
        echo "Not enough memory to run 'run.sh' again."
    fi

    # Wait for a while before checking again
    echo "Waiting for the next check..."
    sleep 10
done
