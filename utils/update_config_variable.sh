# This script updates environment variables
                                                                                                      
# Get user input for variables                                                                        
read -p $'Enter the total number of images in your data set:\n' LAST_IMAGE
((LAST_IMAGE-=1)) # Decrement for 0-indexing
                                                                                                      
# If configuration file exists, prompt user if they want to overwrite                                 
if test -f "config_params.txt"; then                                                                  
  read -p $'A previous configuration exists. Overwrite? Enter [y/n].\n' OVERWRITE                     
  if [ "$OVERWRITE" == "y" ]; then                                                                    
    rm config_params.txt                                                                              
  else                                                                                                
    echo "Configuration cancelled. No files overwritten."                                             
    exit                                                                                              
  fi                                                                                                  
fi                                                                                                    
                                                                                                      
# Write the environment variable                                                                      
echo "export LAST_IMAGE='$LAST_IMAGE'" >> ../config_params.txt                                       
                                                                                                      
# Tell user to source bashrc                                                                          
echo "Please source config_params.txt for changes to take effect."                                    

