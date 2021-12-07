#bash scan_varying_indexer.sh
#cctbx.python sequence_to_stills_no_sb.py dials_temp_files/refined_varying.*
#cctbx.python dials_assign.py
#mkdir dials_temp_files/split_expts
#cd dials_temp_files/split_expts/
#dials.split_experiments ../optimized.*
#cd ..
#mkdir multi_expts
#cd multi_expts
#for i in $(seq -f "%03g" 0 1 178)
#do 
#  cctbx.python ../../assign_beams_to_stills.py ../split_expts/split_${i}.expt ../split_expts/split_${i}.refl multi_${i}
#done
#cd ..
#mkdir refined_expts
#cd refined_expts/
for i in $(seq -f "%03g" 0 1 178)
do 
  bash ../../refine_stills_no_sb.sh multi_${i}.* ultra_refined${i}
done

