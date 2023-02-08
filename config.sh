# UI script for interacting with dataset configurations

# Make temp files directory if it does not exist
if [ ! -d 'dials_temp_files' ]; then
    mkdir dials_temp_files
fi

# Get user input on what to do
read -p $'Enter the number for what you want to do.\n\t 1. Change an environment variable.\n\t 2. Shrink the imageset for a pair of expt, refl files.\n\t 3. Archive the current analysis files.\n\t 4. Split a pair of expt, refl files into a a series of expt, refl files per image.\n' OPTION

# Option 1: Change an environment variable
if [ "$OPTION" == "1"  ]; then
  # If configuration file exists, prompt user if they want to overwrite      
  if test -f "config_params.txt"; then                                    
    read -p $'A previous configuration exists. Delete? Enter [y/n].\n' DELETE
    if [ "$DELETE" == "y" ]; then                                            
      rm config_params.txt                                                
      echo "File deleted. Please initialize a new configuration file to edit variables.\n"                                                 
      exit                                                                   
    else                                                                     
      read -p $'Overwrite configuration file? Enter [y/n]\n' OVERWRITE       
      if [ "OVERWRITE" == "y" ]; then                                        
        # TODO: IMPLEMENT                                                    
      else                                                                   
        echo "Configuration cancelled. No files overwritten. Please archive the previous configuration in order to start a new one.\n"              
        exit                                                                 
      fi                                                                     
    fi                                                                       
  fi                                                                         
  bash utils/update_variables.sh

# Option 2: Shrink the imageset for a pair of expt, refl files
elif [ "$OPTION" = "2" ]; then
  read -p $'Enter the image you\'d like to start with: ' IMG1
  read -p $'Enter the image you\'d like to end with (inclusive!): ' IMG2
  read -p $'Enter the experiment filename: ' -e EXPT
  read -p $'Enter the reflection filename: ' -e REFL
  cctbx.python utils/reset_images.py $IMG1 $IMG2 $EXPT $REFL

# Option 3: Archive data in dials_temp_files
elif [ "$OPTION" = "3" ]; then
  read -p $'Enter the directory you\'d like to save the tarball to:\n' -e TARPATH
  read -p $'Enter a short description for the dataset: \n' DESC
  bash utils/archive_analysis.sh $DESC $TARPATH

# Option 4: Split the imageset for a pair of expt, refl files
elif [ "$OPTION" = "4" ]; then
  read -p $'Enter the experiment filename: ' -e EXPT
  read -p $'Enter the reflection filename: ' -e REFL
  cctbx.python utils/split_by_image.py $EXPT $REFL

# Invalid option
else
  echo "Please enter a valid number."
  exit
fi
