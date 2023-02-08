# Use to define environment variables for location of data

# Get user input for variables
read -p $'Enter the path to the directory containing the diffraction image files:\n' DIFF_IMG_DIR

# Check that directory exists
if [ ! -d "$DIFF_IMG_DIR" ]; then
  echo $DIFF_IMG_DIR "does not exist. Rerun configuration and provide accurate input.\n"
  exit
fi;

# Check that directory contains mccd files
count=$(ls -1 ${DIFF_IMG_DIR}/*.mccd 2>/dev/null | wc -l)
if [ $count == 0 ]; then
  echo "Directory exists but contains no .mccd files. Rerun configuration and provide accurate input.\n"
  exit
fi;

# If configuration file exists, prompt user if they want to overwrite
if test -f "../config_params.txt"; then
  read -p $'A previous configuration exists. Delete? Enter [y/n].\n' DELETE
  if [ "$DELETE" == "y" ]; then
    rm ../config_params.txt
    echo "File deleted.\n"
    exit
  else
    read -p $'Overwrite configuration file? Enter [y/n]\n' OVERWRITE
    if [ "OVERWRITE" == "y" ]; then
      # TODO: IMPLEMENT
    else
      echo "Configuration cancelled. No files overwritten.\n"
      exit
    fi
  fi
fi

# Start configuration file
echo -n "" > config_params.txt

# Write the environment variable
echo "export DIFF_IMG_DIR='$DIFF_IMG_DIR'" >> config_params.txt

# Tell user to source bashrc
echo "Please source config_params.txt for changes to take effect."
