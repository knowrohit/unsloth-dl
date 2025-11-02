#!/bin/bash
# Real-time monitoring script for all 4 training runs

echo "=================================="
echo "Real-time Monitoring - All 4 Runs"
echo "Press Ctrl+C to exit"
echo "=================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if log files exist, if not wait a bit
for i in {1..4}; do
    if [ ! -f "logs/config${i}.log" ]; then
        echo "Waiting for logs/config${i}.log to be created..."
        sleep 2
    fi
done

# Use multi-tail if available, otherwise use a simple approach
if command -v multitail &> /dev/null; then
    # Use multitail for better visualization
    multitail -s 2 \
        -l "tail -f logs/config1.log" \
        -l "tail -f logs/config2.log" \
        -l "tail -f logs/config3.log" \
        -l "tail -f logs/config4.log" \
        -cT ansi
else
    # Fallback: Use simple tail with colors
    echo "Using simple tail (install 'multitail' for better visualization)"
    echo ""
    
    # Use a simple approach with tmux-like splits or just show all logs
    while true; do
        clear
        echo "=================================="
        echo "CONFIG 1 (GPU 0) - r=32, lr=2e-4"
        echo "=================================="
        tail -n 10 logs/config1.log 2>/dev/null || echo "No log yet..."
        
        echo ""
        echo "=================================="
        echo "CONFIG 2 (GPU 1) - r=16, lr=3e-4"
        echo "=================================="
        tail -n 10 logs/config2.log 2>/dev/null || echo "No log yet..."
        
        echo ""
        echo "=================================="
        echo "CONFIG 3 (GPU 2) - r=8, lr=2e-4, seq=512"
        echo "=================================="
        tail -n 10 logs/config3.log 2>/dev/null || echo "No log yet..."
        
        echo ""
        echo "=================================="
        echo "CONFIG 4 (GPU 3) - r=16, lr=1e-4"
        echo "=================================="
        tail -n 10 logs/config4.log 2>/dev/null || echo "No log yet..."
        
        echo ""
        echo "Last updated: $(date '+%H:%M:%S') - Refresh every 3 seconds"
        sleep 3
    done
fi


# Real-time monitoring script for all 4 training runs

echo "=================================="
echo "Real-time Monitoring - All 4 Runs"
echo "Press Ctrl+C to exit"
echo "=================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if log files exist, if not wait a bit
for i in {1..4}; do
    if [ ! -f "logs/config${i}.log" ]; then
        echo "Waiting for logs/config${i}.log to be created..."
        sleep 2
    fi
done

# Use multi-tail if available, otherwise use a simple approach
if command -v multitail &> /dev/null; then
    # Use multitail for better visualization
    multitail -s 2 \
        -l "tail -f logs/config1.log" \
        -l "tail -f logs/config2.log" \
        -l "tail -f logs/config3.log" \
        -l "tail -f logs/config4.log" \
        -cT ansi
else
    # Fallback: Use simple tail with colors
    echo "Using simple tail (install 'multitail' for better visualization)"
    echo ""
    
    # Use a simple approach with tmux-like splits or just show all logs
    while true; do
        clear
        echo "=================================="
        echo "CONFIG 1 (GPU 0) - r=32, lr=2e-4"
        echo "=================================="
        tail -n 10 logs/config1.log 2>/dev/null || echo "No log yet..."
        
        echo ""
        echo "=================================="
        echo "CONFIG 2 (GPU 1) - r=16, lr=3e-4"
        echo "=================================="
        tail -n 10 logs/config2.log 2>/dev/null || echo "No log yet..."
        
        echo ""
        echo "=================================="
        echo "CONFIG 3 (GPU 2) - r=8, lr=2e-4, seq=512"
        echo "=================================="
        tail -n 10 logs/config3.log 2>/dev/null || echo "No log yet..."
        
        echo ""
        echo "=================================="
        echo "CONFIG 4 (GPU 3) - r=16, lr=1e-4"
        echo "=================================="
        tail -n 10 logs/config4.log 2>/dev/null || echo "No log yet..."
        
        echo ""
        echo "Last updated: $(date '+%H:%M:%S') - Refresh every 3 seconds"
        sleep 3
    done
fi

