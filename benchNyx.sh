#!/bin/bash

# Check if the input file path is provided
if [ $# -ne 4 ]; then
  echo "Usage: $0 <path_to_input_file> <dim_x> <dim_y> <dim_z>"
  exit 1
fi

# Input arguments
input_file="$1"
dim_x="$2"
dim_y="$3"
dim_z="$4"

# Calculate dimensionality products
dim1=$((dim_x * dim_y * dim_z))
dim2=$((dim_x * dim_y))

# List of r parameters to iterate over
r_values=(1 2 4 8 16 32)

# List of dimensionality options
dims_list=(
  "-1 $dim1"
  "-2 $dim_x $dim2"
  "-3 $dim_x $dim_y $dim_z"
)

# Base command components
compressed_file="dark_matter_sycl.f32.zfp"
output_file="dark_matter_sycl.f32.zfp.out"
sycl_flag="-x sycl"

# Nested loops: Iterate over both dimensionalities and r parameters
for dims in "${dims_list[@]}"; do
  for r in "${r_values[@]}"; do
    echo "Running with dimensions: $dims and r=$r..."
    ./bin/zfp -f -i "$input_file" -z "$compressed_file" -o "$output_file" $dims -r $r -s $sycl_flag
  done
done