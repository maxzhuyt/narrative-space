#!/bin/bash
# Submit all 13 model continuation jobs to DeltaAI ghx4 partition.
# Usage: bash submit_all.sh

set -u

echo "Submitting all continuation jobs to ghx4 partition..."
echo ""

JOBIDS=""

for sbatch_file in \
    run_qwen25_32b_base.sbatch \
    run_qwen25_32b_instruct.sbatch \
    run_mistral_small_24b_base.sbatch \
    run_mistral_small_24b_instruct.sbatch \
    run_gemma4_31b_base.sbatch \
    run_gemma4_31b_it.sbatch \
    run_llama4_scout_base.sbatch \
    run_llama4_scout_instruct.sbatch \
    run_qwen3_32b_nothinking.sbatch \
    run_qwen3_32b_thinking.sbatch \
    run_qwen35_35b_a3b_nothinking.sbatch \
    run_qwen35_35b_a3b_thinking.sbatch \
    run_qwq_32b.sbatch; do

    if [ ! -f "$sbatch_file" ]; then
        echo "MISSING: $sbatch_file (skipping)"
        continue
    fi

    echo "Submitting: $sbatch_file"
    OUT=$(sbatch "$sbatch_file" 2>&1)
    JOBID=$(echo "$OUT" | awk '{print $4}')
    if [[ "$OUT" == Submitted* ]] && [ -n "$JOBID" ]; then
        echo "  -> Job ID: $JOBID"
        JOBIDS="$JOBIDS $JOBID"
    else
        echo "  -> FAILED to submit: $OUT"
    fi
done

echo ""
echo "All jobs submitted. Job IDs:$JOBIDS"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all with: scancel$JOBIDS"
