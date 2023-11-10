#!/bin/bash
#
cd ../src/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done

cd ../src/criterions/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done

cd ../../src/models/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done

cd ../../src/modules/

filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done

cd ../../src/optim/

filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done

cd ../../src/tasks/

filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done
