#!/bin/bash
rm -rf lightning_logs

APP="toy_example.py"
CMD="python ${APP}"

T0=${SECONDS}
echo "Lightning run started: $(date)"
echo "${CMD}"
${CMD}
echo "Lightning run completed: $(date)"
echo "Run time: $((${SECONDS}-${T0})) seconds"
