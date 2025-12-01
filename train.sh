python train_lightning.py --config configs/transformer.yml --opts \
        DATA.TRAIN_PATH "data/nadi-all/train.txt \
        DATA.TEST_PATH "data/nadi-test/test.txt" \
        MODEL.USE_ASR True \
        MODEL.LOAD_TEXT_BRANCH_ONLY True \
        TRAIN.SAVE_DIR "nadi-results/transformer-text+asr/tashkeela+nadi" \
        MODEL.PRETRAINED_PATH "outputs/results/transformer-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt"