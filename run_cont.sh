#!/bin/bash

# Create a tmux session (session name 'gpumon')
echo "Creating tmux session 'gpumon'..."
tmux new-session -d -s gpumon

# Define a minimum required free GPU memory (in MB) to run 'run.sh'
min_required_gpu_ram=1000  # Adjust this value as needed

# Initialize occupied_ram variable
occupied_ram=0

while true; do
    # Check the available GPU memory
    echo "Checking available GPU memory..."
    gpu_ram_available=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

    echo "Available GPU RAM: $gpu_ram_available MB"

    # Calculate total memory requirement considering the occupied memory
    total_memory_required=$((min_required_gpu_ram + occupied_ram))

    # Check if available GPU memory is greater than the total required memory
    if (( gpu_ram_available > total_memory_required )); then
        echo "Sufficient memory available, running 'run.sh'..."

        # Save the current available memory before running 'run.sh'
        gpu_ram_before_run=$gpu_ram_available

        # Generate a unique identifier for this run
        unique_id=$(date +%s)

        # Create a new window in tmux for running 'run.sh' and execute it
        window_name="run_script_$unique_id"
        tmux new-window -t gpumon -n "$window_name"
        tmux send-keys -t gpumon:"$window_name" './run.sh' C-m

        # Wait for some time to let 'run.sh' start and occupy memory
        sleep 25

        # Check the available GPU memory after running 'run.sh'
        gpu_ram_after_run=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

        # Calculate the memory occupied by 'run.sh'
        run_gpu_occupied=$((gpu_ram_before_run - gpu_ram_after_run))

        # Update the occupied_ram for the next iteration
        occupied_ram=$((occupied_ram + run_gpu_occupied))

        echo "Memory occupied by 'run.sh': $run_gpu_occupied MB"
    else
        echo "Not enough memory to run 'run.sh'."
    fi
done
