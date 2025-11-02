#!/bin/bash
# Enhanced real-time monitoring with wandb links and progress

echo "=================================="
echo "Training Monitor - All 4 Configurations"
echo "=================================="
echo ""

# Function to show wandb links
show_wandb_links() {
    echo "📊 Weights & Biases Dashboard:"
    echo "   https://wandb.ai/projectsolomon/math-verification-multiconfig"
    echo ""
}

# Function to extract key info from logs
extract_progress() {
    local logfile=$1
    local config_name=$2
    
    if [ -f "$logfile" ]; then
        # Get last few lines
        local last_lines=$(tail -n 20 "$logfile" 2>/dev/null)
        
        # Check if training started
        if echo "$last_lines" | grep -q "Starting training"; then
            echo "🟢 Status: Training"
        elif echo "$last_lines" | grep -q "Loading model"; then
            echo "🟡 Status: Loading model..."
        elif echo "$last_lines" | grep -q "Formatting"; then
            echo "🟡 Status: Preparing data..."
        else
            echo "🔵 Status: Initializing..."
        fi
        
        # Extract step/epoch info if available
        if echo "$last_lines" | grep -qE "step|epoch"; then
            local step_info=$(echo "$last_lines" | grep -oE "[0-9]+/[0-9]+" | tail -1)
            if [ ! -z "$step_info" ]; then
                echo "   Progress: $step_info"
            fi
        fi
        
        # Extract loss if available
        if echo "$last_lines" | grep -qE "loss"; then
            local loss=$(echo "$last_lines" | grep -oE "loss.*[0-9]+\.[0-9]+" | tail -1)
            if [ ! -z "$loss" ]; then
                echo "   $loss"
            fi
        fi
    else
        echo "⚪ Status: Waiting for log file..."
    fi
}

# Main monitoring loop
while true; do
    clear
    echo "=================================="
    echo "  TRAINING MONITOR - All 4 Runs"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================="
    echo ""
    
    show_wandb_links
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ CONFIG 1: LoRA r=32, LR=2e-4, seq=1024 (GPU 0)              │"
    echo "└─────────────────────────────────────────────────────────────┘"
    extract_progress "logs/config1.log" "config1"
    echo ""
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ CONFIG 2: LoRA r=16, LR=3e-4, seq=1024 (GPU 1)              │"
    echo "└─────────────────────────────────────────────────────────────┘"
    extract_progress "logs/config2.log" "config2"
    echo ""
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ CONFIG 3: LoRA r=8, LR=2e-4, seq=512 (GPU 2)                 │"
    echo "└─────────────────────────────────────────────────────────────┘"
    extract_progress "logs/config3.log" "config3"
    echo ""
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ CONFIG 4: LoRA r=16, LR=1e-4, seq=1024 (GPU 3)              │"
    echo "└─────────────────────────────────────────────────────────────┘"
    extract_progress "logs/config4.log" "config4"
    echo ""
    
    echo "=================================="
    echo "Press Ctrl+C to exit monitor"
    echo "Refresh: Every 5 seconds"
    echo "=================================="
    
    sleep 5
done


# Enhanced real-time monitoring with wandb links and progress

echo "=================================="
echo "Training Monitor - All 4 Configurations"
echo "=================================="
echo ""

# Function to show wandb links
show_wandb_links() {
    echo "📊 Weights & Biases Dashboard:"
    echo "   https://wandb.ai/projectsolomon/math-verification-multiconfig"
    echo ""
}

# Function to extract key info from logs
extract_progress() {
    local logfile=$1
    local config_name=$2
    
    if [ -f "$logfile" ]; then
        # Get last few lines
        local last_lines=$(tail -n 20 "$logfile" 2>/dev/null)
        
        # Check if training started
        if echo "$last_lines" | grep -q "Starting training"; then
            echo "🟢 Status: Training"
        elif echo "$last_lines" | grep -q "Loading model"; then
            echo "🟡 Status: Loading model..."
        elif echo "$last_lines" | grep -q "Formatting"; then
            echo "🟡 Status: Preparing data..."
        else
            echo "🔵 Status: Initializing..."
        fi
        
        # Extract step/epoch info if available
        if echo "$last_lines" | grep -qE "step|epoch"; then
            local step_info=$(echo "$last_lines" | grep -oE "[0-9]+/[0-9]+" | tail -1)
            if [ ! -z "$step_info" ]; then
                echo "   Progress: $step_info"
            fi
        fi
        
        # Extract loss if available
        if echo "$last_lines" | grep -qE "loss"; then
            local loss=$(echo "$last_lines" | grep -oE "loss.*[0-9]+\.[0-9]+" | tail -1)
            if [ ! -z "$loss" ]; then
                echo "   $loss"
            fi
        fi
    else
        echo "⚪ Status: Waiting for log file..."
    fi
}

# Main monitoring loop
while true; do
    clear
    echo "=================================="
    echo "  TRAINING MONITOR - All 4 Runs"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================="
    echo ""
    
    show_wandb_links
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ CONFIG 1: LoRA r=32, LR=2e-4, seq=1024 (GPU 0)              │"
    echo "└─────────────────────────────────────────────────────────────┘"
    extract_progress "logs/config1.log" "config1"
    echo ""
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ CONFIG 2: LoRA r=16, LR=3e-4, seq=1024 (GPU 1)              │"
    echo "└─────────────────────────────────────────────────────────────┘"
    extract_progress "logs/config2.log" "config2"
    echo ""
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ CONFIG 3: LoRA r=8, LR=2e-4, seq=512 (GPU 2)                 │"
    echo "└─────────────────────────────────────────────────────────────┘"
    extract_progress "logs/config3.log" "config3"
    echo ""
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ CONFIG 4: LoRA r=16, LR=1e-4, seq=1024 (GPU 3)              │"
    echo "└─────────────────────────────────────────────────────────────┘"
    extract_progress "logs/config4.log" "config4"
    echo ""
    
    echo "=================================="
    echo "Press Ctrl+C to exit monitor"
    echo "Refresh: Every 5 seconds"
    echo "=================================="
    
    sleep 5
done

