import numpy as np
from dxtbx.model import ExperimentList
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from IPython import embed

# expt_file = "dials_temp_files/mega_ultra_refined.expt"
# refl_file = "dials_temp_files/mega_ultra_refined.refl"
# elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
# refls = reflection_table.from_file(refl_file)

def gen_experiment_identifiers(refls, elist):
    """Generates a mapping of the ID column in a reflection table 
    to the identifiers in an experiment list. The values of the ID
    column in the reflection table are taken to be the indices of
    the identifiers in order of the experiment list."""
    # Delete old mapping
    for k in refls.experiment_identifiers().keys():
        del refls.experiment_identifiers()[k]

    # Make arrays for keys and values
    indices = refls['id'].as_numpy_array()
    identifiers = np.empty_like(indices, dtype=np.dtype('U12'))

    # Populate identifiers based on indices
    for i, j in enumerate(indices):
        identifiers[i] = str(j)
        refls.experiment_identifiers()[int(j)] = identifiers[i]

    return refls
