out=merge
mtz=integrated.mtz
mkdir $out

BASE_ARGS=(
  --iterations=30_000 
  --gpu-id=0 
  --refine-uncertainties 
  --merge-half-datasets
  --wavelength-key='wavelength' 
  "xobs,yobs,wavelength,dHKL,BATCH" 
)


careless poly \
  ${BASE_ARGS[@]} \
  $mtz \
  $out/dhfr
