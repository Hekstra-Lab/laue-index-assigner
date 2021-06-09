# Use to define environment variables for location of data

# Get user input for variables
read -p $'Enter the path to the directory containing .mccd files for DHFR:\n' DHFRMCCDDIR

# Check that directory exists
if [ ! -d "$DHFRMCCDDIR" ]; then
  echo $DHFRMCCDDIR "does not exist. Rerun configuration and provide accurate input."
  exit
fi;

# Check that directory contains mccd files
count=$(ls -1 ${DHFRMCCDDIR}/*.mccd 2>/dev/null | wc -l)
if [ $count == 0 ]; then
  echo "Directory exists but contains no .mccd files. Rerun configuration and provide accurate input."
  exit
fi;

# Export environment variables
echo "export DHFRMCCDDIR=$DHFRMCCDDIR" >> ~/.bashrc
