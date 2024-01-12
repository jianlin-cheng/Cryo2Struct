# Execute data preparation script
echo "Script started for data preparation"
relative_path=$1
density_map_dir=$(readlink -f "$relative_path")
for i in "${density_map_dir[@]}"; do
  echo "Running for density maps present in : $i"
  python3 preprocess/get_resample_map.py "$i"
  python3 preprocess/get_normalize_map.py "$i"
  wait
  echo "Done with maps in directory: $i"
  echo "ALL DONE!" 
done
