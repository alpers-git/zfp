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
dim2=$((dim_y * dim_z))

# Validate file extension and set mode (-f or -d) and r_values
if [[ "$input_file" == *.f32 ]]; then
  mode_flag="-f"
  r_values=(1 2 4 8 16 32)
elif [[ "$input_file" == *.d64 ]]; then
  mode_flag="-d"
  r_values=(1 2 4 8 16 32 64)
else
  echo "Error: Input file must have a .f32 or .d64 extension."
  exit 1
fi

# List of dimensionality options
dims_list=(
  "-1 $dim1"
  "-2 $dim_x $dim2"
  "-3 $dim_x $dim_y $dim_z"
)

# Base command components
compressed_file="compressed.f32.zfp"
output_file="uncompressed.f32.zfp.out"
sycl_flag="-x sycl"

# Nested loops: Iterate over both dimensionalities and r parameters
for dims in "${dims_list[@]}"; do
  for r in "${r_values[@]}"; do
    echo "Running with dimensions: $dims and r=$r..."
    ./bin/zfp $mode_flag -i "$input_file" -z "$compressed_file" -o "$output_file" $dims -r $r -s $sycl_flag
  done
done