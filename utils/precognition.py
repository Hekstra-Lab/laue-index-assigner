from tqdm import tqdm
import numpy as np
import reciprocalspaceship as rs


def parse_ii_inp_file_pairs(ii_filenames, inp_filenames, spacegroup=None, log=None):
    data = None
    sg_number = None
    p = []
    cell = np.zeros(6)
    for file_number, (iiFN,inpFN) in enumerate(tqdm(list(zip(ii_filenames, inp_filenames)))):
        df = rs.read_precognition(iiFN)
        with open(inpFN) as f:
            for line in f:
                if "Crystal" in line:
                    cell +=  np.array(list(map(float, line.split()[1:7])))/len(ii_filenames)
                    if spacegroup is None:
                        sg_number  = int(line.split()[7])
                if "Pixel" in line:
                    pixel_x = float(line.split()[1])
                    pixel_y = float(line.split()[2])
        del(df['Resolution'])

        # If log is given, use it to determine the image number
        if log is None:
            df['BATCH'] = file_number
        else:
            entry = log.loc[log.file == inpFN[:-4]]
            assert len(entry) == 1
            df['BATCH'] = entry.index.values[0]

        #Purge multiples from Precognition processing
        #These will be recomputed during merging later
        df = df.reset_index().groupby(['X', 'Y'], as_index=False).first()
        df.cell = rs.dataset.gemmi.UnitCell(*cell)
        if sg_number is not None:
            df.spacegroup = rs.dataset.gemmi.find_spacegroup_by_number(sg_number)
        data = df.append(data)

    del(data['Multiplicity'])
    data.set_index(['H', 'K', 'L'], inplace=True)
    data.infer_mtz_dtypes(inplace=True)
    return data


