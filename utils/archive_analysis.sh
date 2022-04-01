# Script for archiving an analysis
set -x

# Check that output path is a legitimate directory
if [ ! -d "`eval echo ${2//>}`" ]; then
  echo "Directory does not exist. Make directory and then execute again."
  exit
fi

# Check that analysis files not empty
if [ -z "$(ls -A dials_temp_files)" ]; then
  echo 'No analysis files to archive in dials_temp_files. Exiting.'
  exit
fi

# Write description file
echo $1 >> dials_temp_files/DESCRIPTION.txt

# Tarball the analysis files
tar -cvf analysis.tar dials_temp_files/*

# Move tarball to output directory
mv -v analysis.tar `eval echo ${2//>}`

# Empty directory
rm dials_temp_files/*
