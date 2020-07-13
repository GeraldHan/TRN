CUDA_VISIBLE_DEVICES=7 python validate.py --mode test \
--input_dir process \
--ckpt trainval_dfaftime/model.pt \
--output_file test_std_dfaftime.json \
--test_question_pt new_2/test_std_questions.pt \
--test_question_json data/v2_OpenEnded_mscoco_test2015_questions.json

CUDA_VISIBLE_DEVICES=7 python validate.py --mode val \
--input_dir process \
--ckpt trainval_dfaftime/model.pt \