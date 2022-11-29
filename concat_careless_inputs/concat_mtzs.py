import reciprocalspaceship as rs
import pandas as pd
import glob

def read_concat_write(mtzinlist, mtzout):
    """
    This function will:
     - concatenate the input mtzs
     - use the cell of the first mtz provide 
     - remove systematic absences from the parent spacegroup
     - after removing absences, cooerce the spacegroup to P1
     - write out the concatenated, P1 mtz
    """

    out = pd.concat(mtzinlist[:2])

    if len(mtzinlist) > 2:
        for mtz in mtzinlist[2:]:
            out = pd.concat([out, mtz])

    out.cell = mtzinlist[0].cell

    # temporarily set spacegroup so that rs knows about absences
    out.spacegroup = mtzinlist[0].spacegroup
    print(out.spacegroup, len(out))
    out.remove_absences(inplace=True)

    # cooerce to P1
    out.spacegroup = 'P1'
    print(out.spacegroup, len(out))

    out.write_mtz(mtzout)

    return

read_concat_write(
    [rs.read_mtz(file) for file in glob.glob('../dials_files_200ns_*/integrated_from_int_test.mtz')],
    'concat_200ns_p1_int_test.mtz'
    )

read_concat_write(
    [rs.read_mtz(file) for file in glob.glob('../dials_files_200ns_*/integrated_from_integrate.mtz')],
    'concat_200ns_p1_integrate.mtz'
    )

read_concat_write(
    [rs.read_mtz(file) for file in glob.glob('../dials_files_640ns_*/integrated_from_int_test.mtz')],
    'concat_640ns_p1_int_test.mtz'
    )

read_concat_write(
    [rs.read_mtz(file) for file in glob.glob('../dials_files_640ns_*/integrated_from_integrate.mtz')],
    'concat_640ns_p1_integrate.mtz'
    )

read_concat_write(
    [rs.read_mtz(file) for file in glob.glob('../dials_files_off_*/integrated_from_integrate.mtz')],
    'concat_off_p1_integrate.mtz'
    )
