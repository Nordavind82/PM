#!/bin/bash
# Monitored PSTM execution script
# Runs bins in small batches with fresh Python processes to avoid OOM
# Exports to SEG-Y after completion

LOG_FILE="/Users/olegadamovich/pstm/pstm_monitor.log"
BATCH_SIZE=5
MAX_RETRIES=3
TOTAL_BINS=40

echo "========================================" | tee "$LOG_FILE"
echo "PSTM Monitored Execution Started" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Batch size: $BATCH_SIZE bins per Python process" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd /Users/olegadamovich/pstm

# Function to count completed bins
count_completed() {
    ls /Users/olegadamovich/SeismicData/PSTM_common_offset/ 2>/dev/null | grep -c migration_bin || echo 0
}

# Function to get list of incomplete bins
get_incomplete_bins() {
    local incomplete=""
    for i in $(seq 0 39); do
        bin_dir=$(printf "/Users/olegadamovich/SeismicData/PSTM_common_offset/migration_bin_%02d" $i)
        if [ ! -d "$bin_dir" ] || [ ! -d "$bin_dir/migrated_stack.zarr" ]; then
            if [ -z "$incomplete" ]; then
                incomplete="$i"
            else
                incomplete="$incomplete,$i"
            fi
        fi
    done
    echo "$incomplete"
}

# Main loop - process in batches
while true; do
    COMPLETED=$(count_completed)
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date)] Progress: $COMPLETED / $TOTAL_BINS bins completed" | tee -a "$LOG_FILE"

    if [ "$COMPLETED" -ge "$TOTAL_BINS" ]; then
        echo "[$(date)] All bins completed!" | tee -a "$LOG_FILE"
        break
    fi

    # Get incomplete bins
    INCOMPLETE=$(get_incomplete_bins)

    if [ -z "$INCOMPLETE" ]; then
        echo "[$(date)] No incomplete bins found, all done!" | tee -a "$LOG_FILE"
        break
    fi

    # Take first BATCH_SIZE bins from incomplete list
    BATCH=$(echo "$INCOMPLETE" | tr ',' '\n' | head -n $BATCH_SIZE | tr '\n' ',' | sed 's/,$//')

    echo "[$(date)] Processing batch: bins $BATCH" | tee -a "$LOG_FILE"
    echo "[$(date)] Starting fresh Python process..." | tee -a "$LOG_FILE"

    # Run batch with retries
    RETRY=0
    SUCCESS=false

    while [ $RETRY -lt $MAX_RETRIES ] && [ "$SUCCESS" = "false" ]; do
        if [ $RETRY -gt 0 ]; then
            echo "[$(date)] Retry $RETRY/$MAX_RETRIES for batch $BATCH" | tee -a "$LOG_FILE"
            sleep 30
        fi

        # Run migration for this batch in a fresh Python process
        python3 run_pstm_all_offsets.py --bins "$BATCH" --skip-velocity-check 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            SUCCESS=true
            echo "[$(date)] Batch $BATCH completed successfully" | tee -a "$LOG_FILE"
        else
            RETRY=$((RETRY + 1))
            echo "[$(date)] Batch failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"

            # Force memory cleanup
            echo "[$(date)] Waiting for memory cleanup..." | tee -a "$LOG_FILE"
            sleep 10
        fi
    done

    if [ "$SUCCESS" = "false" ]; then
        echo "[$(date)] Batch $BATCH failed after $MAX_RETRIES retries, moving to next batch" | tee -a "$LOG_FILE"
    fi

    # Memory cleanup between batches
    echo "[$(date)] Batch done, cleaning up memory before next batch..." | tee -a "$LOG_FILE"
    sleep 5

    # Show current progress
    COMPLETED=$(count_completed)
    REMAINING=$((TOTAL_BINS - COMPLETED))
    echo "[$(date)] Status: $COMPLETED completed, $REMAINING remaining" | tee -a "$LOG_FILE"
done

# Final count
COMPLETED=$(count_completed)
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "PSTM MIGRATION COMPLETED" | tee -a "$LOG_FILE"
echo "Total bins completed: $COMPLETED / $TOTAL_BINS" | tee -a "$LOG_FILE"
echo "Migration end time: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Export to SEG-Y
if [ "$COMPLETED" -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Starting SEG-Y Export..." | tee -a "$LOG_FILE"
    echo "Export start time: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    python3 export_to_segy.py 2>&1 | tee -a "$LOG_FILE"
    EXPORT_EXIT=$?

    if [ $EXPORT_EXIT -eq 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "SEG-Y EXPORT COMPLETED SUCCESSFULLY!" | tee -a "$LOG_FILE"
        echo "Export end time: $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
    else
        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "SEG-Y EXPORT FAILED with exit code $EXPORT_EXIT" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
    fi
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "ALL TASKS COMPLETED!" | tee -a "$LOG_FILE"
echo "Final end time: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
