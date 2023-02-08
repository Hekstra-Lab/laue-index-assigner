# Script to combine integrated stills into one mtz file

from tqdm import trange
import reciprocalspaceship as rs

# Open first still
total_integrated_mtz = rs.read_mtz('/n/home04/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/integration/integrated000000.mtz')

# Append all other stills
for i in trange(1, 3049):
    try:
        filename = f'/n/home04/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/integration/integrated{i:06d}.mtz'
        img_mtz = rs.read_mtz(filename)
        img_mtz['BATCH'] = i+1
        total_integrated_mtz = total_integrated_mtz.append(img_mtz)
    except:
        print(f'Image {i:06d} could not be integrated.')
        continue

# Fix data type of BATCH
total_integrated_mtz['BATCH'] = total_integrated_mtz['BATCH'].infer_mtz_dtype()

# Write output
total_integrated_mtz.write_mtz('dials_temp_files/integration/hewl_integrated.mtz')
