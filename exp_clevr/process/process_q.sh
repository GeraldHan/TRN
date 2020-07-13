python preprocess_questions.py --input_questions_json ../data/CLEVR_v1.0/questions/CLEVR_train_questions.json \
--output_pt_file ../input/train_questions.pt \
--output_vocab_json ../input/vocab.json  

python preprocess_questions.py --input_questions_json ../data/CLEVR_v1.0/questions/CLEVR_val_questions.json \
--output_pt_file ../input/val_questions.pt \
--input_vocab_json ../input/vocab.json

python preprocess_questions.py --input_questions_json ../data/CLEVR_v1.0/questions/CLEVR_test_questions.json \
--output_pt_file ../input/test_questions.pt \
--input_vocab_json ../input/vocab.json

python preprocess_questions.py --input_questions_json ../data/CLEVR-Humans/CLEVR-Humans-train.json \
--output_pt_file ../input_human/train_questions.pt \
--input_vocab_json ../input/vocab.json --expand_vocab 1
--output_vocab_json ../input_human/vocab.json  

python preprocess_questions.py --input_questions_json ../data/CLEVR-Humans/CLEVR-Humans-val.json \
--output_pt_file ../input_human/val_questions.pt \
--input_vocab_json ../input_human/vocab.json

python preprocess_questions.py --input_questions_json ../data/CLEVR-Humans/CLEVR-Humans-test.json \
--output_pt_file ../input_human/test_questions.pt \
--input_vocab_json ../input_human/vocab.json