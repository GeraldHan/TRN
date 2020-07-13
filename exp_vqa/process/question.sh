python process_q.py --input_questions_json ../data/v2_OpenEnded_mscoco_trainval2014_questions.json \
--input_annotations_json ../data/v2_mscoco_trainval2014_annotations.json \
--vocab_json vocab.json --output_pt trainval_questions.pt \
--mode train

python process_q.py --input_questions_json ../data/v2_OpenEnded_mscoco_train2014_questions.json \
--input_annotations_json ../data/v2_mscoco_train2014_annotations.json \
--vocab_json vocab.json --output_pt train_questions.pt \
--mode val

python process_q.py --input_questions_json ../data/v2_OpenEnded_mscoco_val2014_questions.json \
--input_annotations_json ../data/v2_mscoco_val2014_annotations.json \
--vocab_json vocab.json --output_pt val_questions.pt \
--mode val

python process_q.py --input_questions_json ../data/v2_OpenEnded_mscoco_test2015_questions.json \
--vocab_json vocab.json --output_pt test_questions.pt \
--mode test
