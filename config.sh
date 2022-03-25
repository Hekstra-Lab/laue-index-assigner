# UI script for interacting with dataset configurations

# Get user input on what to do
read -p $'Enter the number for what you want to do.\n\t 1. Set the raw data directory.\n\t 2. Shrink the imageset for a pair of expt, refl files.\n\t 3. Archive the current analysis files.\n' OPTION

# Option 1: Set raw data directory
if [ "$OPTION" == "1"  ]; then
  bash utils/raw_data.sh

# Option 2: Shrink the imageset for a pair of expt, refl files
elif [ "$OPTION" = "2" ]; then
  read -p $'Enter the image you\'d like to start with: ' IMG1
  read -p $'Enter the image you\'d like to end with (inclusive!): ' IMG2
  read -p $'Enter the experiment filename: ' EXPT
  read -p $'Enter the reflection filename: ' REFL
  cctbx.python utils/reset_images.py $IMG1 $IMG2 $EXPT $REFL

# Option 3: Archive data in dials_temp_files
elif [ "$OPTION" = "3" ]; then
  read -p $'Enter the path you\'d like to save the tarball to:\n' $TARPATH
  read -p $'Enter a short description for the dataset: \n' $DESC
  bash utils/archive_analysis.sh $DESC $TARPATH

# Invalid option
else
  echo "Please enter a valid number."
  exit
fi
