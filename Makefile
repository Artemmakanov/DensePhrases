train-rc: model-name nq-rc-data nq-param
	python train_rc.py \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--data_dir $(DATA_DIR)/single-qa \
		--cache_dir $(CACHE_DIR) \
		--train_file $(TRAIN_DATA) \
		--predict_file $(DEV_DATA) \
		--do_train \
		--do_eval \
		--fp16 \
		--per_gpu_train_batch_size $(BS) \
		--learning_rate $(LR) \
		--num_train_epochs 2.0 \
		--max_seq_length $(MAX_SEQ_LEN) \
		--lambda_kl $(LAMBDA_KL) \
		--lambda_neg $(LAMBDA_NEG) \
		--lambda_flt 1.0 \
		--filter_threshold -2.0 \
		--append_title \
		--evaluate_during_training \
		--teacher_dir $(SAVE_DIR)/$(TEACHER_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--overwrite_output_dir \
		$(OPTIONS)