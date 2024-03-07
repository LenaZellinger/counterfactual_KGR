#!/bin/bash

 declare -a sizes=("s" "l")
  for size in "${sizes[@]}"; do
    python data_generation/generate_data_rule_based.py --size=${size} --save='True'
    python data_generation/create_additional_negatives.py --size=${size} --save='True' --generate='near'
    python data_generation/split_into_valid_and_test.py --size=${size} --save='True'
    python data_generation/verbalize_dataset.py --size=${size} --save='True'
  done

  python data_generation/generate_M/generate_data_rule_based.py --size='m' --save='True'
  python data_generation/create_additional_negatives.py --generate='far' --size='m' --save='True'
  python data_generation/create_additional_negatives.py --generate='near' --size='m' --save='True'
  python data_generation/split_into_valid_and_test.py --size='m' --save='True'
  python data_generation/verbalize_dataset.py --size='m' --save='True'