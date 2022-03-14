# Use to define environment variables for location of data

# Get user input for variables
read -p $'Enter the path to the directory containing the diffraction image files:\n' DIFF_IMG_DIR

# Check that directory exists
if [ ! -d "$DIFF_IMG_DIR" ]; then
  echo $DIFF_IMG_DIR "does not exist. Rerun configuration and provide accurate input."
  exit
fi;

# Check that directory contains mccd files
count=$(ls -1 ${DIFF_IMG_DIR}/*.mccd 2>/dev/null | wc -l)
if [ $count == 0 ]; then
  echo "Directory exists but contains no .mccd files. Rerun configuration and provide accurate input."
  exit
fi;

# Check if bashrc already contains this env variable


# Export environment variables
echo "export DIFF_IMG_DIR='$DIFF_IMG_DIR'" >> ~/.bashrc

# Tell user to source bashrc
echo "Please reload your shell for changes to take effect."
