#!/bin/bash

n_repeats=30

common_args="--margin 15 $TEGO_ARGS"

# array of model sizes
model_sizes=( s m l )

n_objects=( 2 4 9 19 )

number_of_training_examples=( 1 3 5 10 )


for model_size in "${model_sizes[@]}"
do
    for n_object in "${n_objects[@]}"
    do
        for n_training_examples in "${number_of_training_examples[@]}"
        do
            echo "model_size: $model_size, n_object: $n_object, n_training_examples: $n_training_examples"
            python3 main.py --yolo-size $model_size -n $n_object -k $n_training_examples -r $n_repeats $common_args $@
        done
    done
done
echo "All experiments done!"
