nthreads=151
memory=4
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="run_knn.py"

rmin=0.01
rmax=100
nneighbours=8
nsamples=100000000
batch_size=1000000
neval=10000

pythoncm="$env $file --rmin $rmin --rmax $rmax --nneighbours $nneighbours --nsamples $nsamples --batch_size $batch_size --neval $neval"

# echo $pythoncm
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm
