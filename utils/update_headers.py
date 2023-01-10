from marccd import MarCCD
import argparse
import glob
from tqdm import trange

# Get I/O options from user
parser = argparse.ArgumentParser()
parser.add_argument('filenames', help='MCCD files to edit. Enclosed in quotes.', type=str)
parser.add_argument('wavelength', help='Wavelength to use for data.', type=float)
args = parser.parse_args()

# Sort files in order
files = sorted(glob.glob(args.filenames))

# Open files and write appropriate header data
for i in trange(len(files)):
    mccd = files[i]
    try:
        img = MarCCD(mccd)
        img.wavelength = args.wavelength
        img.write(mccd)
    except:
        print(f'Failed reading {files[i]}.')
        continue
