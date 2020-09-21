#!/usr/bin/env bash
python src/main.py test \
--dataset ptb \
--embedding-path data/glove.gz \
--synconst-test-ptb-path data/23.auto.clean \
--syndep-test-ptb-path data/ptb_test_3.3.0.sd \
--model-path-base models/joint.pt