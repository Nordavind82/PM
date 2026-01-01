#!/bin/bash
# Monitored PSTM execution script for 20m offset bins
# Runs bins in small batches with fresh Python processes to avoid OOM
# Exports to SEG-Y after completion

# =============================================================================
# Configuration
# =============================================================================
INPUT_DIR="/Users/olegadamovich/SeismicData/common_offset_20m"
OUTPUT_DIR="/Users/olegadamovich/SeismicData/PSTM_common_offset_20m"
VELOCITY_PATH="/Users/olegadamovich/SeismicData/common_offset_20m/velocity_pstm.zarr"
LOG_FILE="/Users/olegadamovich/pstm/pstm_monitor_20m.log"

BATCH_SIZE=5          # Bins per Python process
MAX_RETRIES=3         # Retries for failed batches
MIN_TRACES=100        # Skip bins with fewer traces

# =============================================================================
# Script Start
# =============================================================================
cd /Users/olegadamovich/pstm

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Auto-detect available bins (count directories only)
AVAILABLE_BINS=0
for dir in "$INPUT_DIR"/offset_bin_*/; do
    if [ -d "$dir" ]; then
        AVAILABLE_BINS=$((AVAILABLE_BINS + 1))
    fi
done

echo "========================================" | tee "$LOG_FILE"
echo "PSTM Monitored Execution (20m bins)" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Input:  $INPUT_DIR" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Available input bins: $AVAILABLE_BINS" | tee -a "$LOG_FILE"
echo "Batch size: $BATCH_SIZE bins per Python process" | tee -a "$LOG_FILE"
echo "Min traces: $MIN_TRACES" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Check if any bins are available
if [ "$AVAILABLE_BINS" -eq 0 ]; then
    echo "ERROR: No input bins found in $INPUT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

# =============================================================================
# Helper Functions
# =============================================================================

# Count completed bins (only count bins that have migrated_stack.zarr)
count_completed() {
    find "$OUTPUT_DIR" -maxdepth 2 -type d -name "migrated_stack.zarr" 2>/dev/null | wc -l | tr -d ' '
}

# Get list of incomplete bins (only from available input bins)
get_incomplete_bins() {
    local incomplete=""

    for input_dir in "$INPUT_DIR"/offset_bin_*/; do
        if [ ! -d "$input_dir" ]; then
            continue
        fi

        # Extract bin number
        bin_name=$(basename "$input_dir")
        bin_num=$(echo "$bin_name" | sed 's/offset_bin_//' | sed 's/^0*//')

        # Skip if no bin number extracted
        if [ -z "$bin_num" ]; then
            continue
        fi

        # Check if input has traces.zarr and headers.parquet
        if [ ! -d "$input_dir/traces.zarr" ] || [ ! -f "$input_dir/headers.parquet" ]; then
            continue  # Skip bins without proper input data
        fi

        # Check if migration output exists and is complete
        output_bin_dir=$(printf "$OUTPUT_DIR/migration_bin_%02d" "$bin_num")
        if [ ! -d "$output_bin_dir/migrated_stack.zarr" ]; then
            if [ -z "$incomplete" ]; then
                incomplete="$bin_num"
            else
                incomplete="$incomplete,$bin_num"
            fi
        fi
    done
    echo "$incomplete"
}

# Get total bins to process (count available input bins with proper data)
get_total_to_process() {
    local count=0

    for input_dir in "$INPUT_DIR"/offset_bin_*/; do
        if [ ! -d "$input_dir" ]; then
            continue
        fi

        if [ -d "$input_dir/traces.zarr" ] && [ -f "$input_dir/headers.parquet" ]; then
            count=$((count + 1))
        fi
    done
    echo $count
}

# =============================================================================
# Main Loop
# =============================================================================
TOTAL_TO_PROCESS=$(get_total_to_process)

while true; do
    COMPLETED=$(count_completed)
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date)] Progress: $COMPLETED / $TOTAL_TO_PROCESS bins completed" | tee -a "$LOG_FILE"

    if [ "$COMPLETED" -ge "$TOTAL_TO_PROCESS" ]; then
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
        python3 run_pstm_all_offsets.py \
            --input-dir "$INPUT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --velocity "$VELOCITY_PATH" \
            --bins "$BATCH" \
            --min-traces $MIN_TRACES \
            --skip-velocity-check 2>&1 | tee -a "$LOG_FILE"
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
    REMAINING=$((TOTAL_TO_PROCESS - COMPLETED))
    echo "[$(date)] Status: $COMPLETED completed, $REMAINING remaining" | tee -a "$LOG_FILE"
done

# =============================================================================
# Final Summary
# =============================================================================
COMPLETED=$(count_completed)
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "PSTM MIGRATION COMPLETED" | tee -a "$LOG_FILE"
echo "Total bins completed: $COMPLETED / $TOTAL_TO_PROCESS" | tee -a "$LOG_FILE"
echo "Migration end time: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# =============================================================================
# SEG-Y Export
# =============================================================================
if [ "$COMPLETED" -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Starting SEG-Y Export..." | tee -a "$LOG_FILE"
    echo "Export start time: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    python3 export_migration_segy.py \
        --migration-dir "$OUTPUT_DIR" \
        --output-dir "$OUTPUT_DIR/segy_export" \
        --bins all 2>&1 | tee -a "$LOG_FILE"
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
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
