#!/bin/bash

# This script sets environment variables to emulate DP support.
export IGC_EnableDPEmulation=1
export SYCL_DEVICE_WHITE_LIST=""
export OverrideDefaultFP64Settings=1