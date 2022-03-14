# Use to define environment variables for location of data

# Get user input for variables
read -p $'Enter the path to the directory containing the diffraction image files:\n' IMG_DIR

# Check that directory exists
if [ ! -d "$IMG_DIR" ]; then
  echo $IMG_DIR "does not exist. Rerun configuration and provide accurate input."
  exit
fi;

# Check that directory contains mccd files
count=$(ls -1 ${IMG_DIR}/*.mccd 2>/dev/null | wc -l)
if [ $count == 0 ]; then
  echo "Directory exists but contains no .mccd files. Rerun configuration and provide accurate input."
  exit
fi;

# Export environment variables
echo "export IMG_DIR=$IMG_DIR" >> ~/.bashrc
