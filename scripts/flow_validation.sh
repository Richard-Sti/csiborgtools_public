memory=4
on_login=0
nthreads=${1}
ksmooth=${2}

queue="berg"
env="/mnt/users/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="flow_validation.py"

#"Pantheon+_zSN"
catalogue="Pantheon+_groups"
simname="Carrick2015"


pythoncm="$env $file --catalogue $catalogue --simname $simname --ksmooth $ksmooth"
if [ $on_login -eq 1 ]; then
    echo $pythoncm
    $pythoncm
else
    cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
    echo "Submitting:"
    echo $cm
    echo
    eval $cm
fi
