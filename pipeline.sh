# Load modules and DIALS
module load gcc/12.1.0-fasrc01
source config_params.txt

# Monochromatic portion
bash scan_varying_indexer.sh

# Split into single-image still files
cctbx.python sequence_to_stills_no_sb.py dials_temp_files/refined_varying.*


# Polychromatic portion
N=64 # Max multiprocessing
for i in $(seq -f "%06g" 0 ${LAST_IMAGE})
do
  ((j=j%N)); ((j++==0)) && wait; 
  (echo "Analyzing image ${i}."; 
  cctbx.python dials_assign.py $i;
  cctbx.python assign_beams_to_stills.py "dials_temp_files/assignment/optimized${i}.expt" "dials_temp_files/assignment/optimized${i}.refl" "dials_temp_files/multibeam/multi${i}";
  bash refine_remove_outliers.sh multi${i}.expt multi${i}.refl mega_ultra_refined${i};
  cctbx.python store_wavelengths.py $i;
  cctbx.python poly_predictor.py "dials_temp_files/refinement/mega_ultra_refined${i}.expt" "dials_temp_files/refinement/mega_ultra_refined${i}.refl" "dials_temp_files/prediction/predicted${i}.refl";
  cctbx.python integrate.py "dials_temp_files/refinement/mega_ultra_refined${i}.expt" "dials_temp_files/prediction/predicted${i}.refl" "dials_temp_files/integration/integrated${i}.mtz";
  cctbx.python int_test.py "dials_temp_files/refinement/mega_ultra_refined${i}.expt" "dials_temp_files/refinement/mega_ultra_refined${i}.refl" "dials_temp_files/prediction/predicted${i}.refl" "dials_temp_files/integration/integrated_dials${i}.expt" "dials_temp_files/integration/integrated_dials${i}.refl";) &
done

cctbx.python combine_integrations.py
#bash hewl_run_careless.sh
#load_phenix
#phenix.refine phenix.eff


