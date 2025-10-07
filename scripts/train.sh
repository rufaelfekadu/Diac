# train lstm models

model=lstm
for model in "lstm" "transformer"
do
    echo "Training model: $model"
    # text only
    python train_lightning.py --config configs/${model}.yml --opts \
        DATA.TRAIN_PATH 'data/tashkeela/train.txt' \
        DATA.VAL_PATH 'data/tashkeela/val.txt' \
        MODEL.USE_ASR False \
        TRAIN.SAVE_DIR 'results/'${model}'-text-only/tashkeela'

    python train_lightning.py --config configs/${model}.yml --opts \
        DATA.TRAIN_PATH 'data/clartts/train_no_special.txt' \
        MODEL.USE_ASR False \
        TRAIN.SAVE_DIR 'results/'${model}'-text-only/clartts'

    python train_lightning.py --config configs/${model}.yml --opts \
        DATA.TRAIN_PATH 'data/clartts/train_no_special.txt' \
        MODEL.USE_ASR False \
        MODEL.PRETRAINED_PATH 'results/'${model}'-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt' \
        MODEL.LOAD_TEXT_BRANCH_ONLY True \
        TRAIN.SAVE_DIR 'results/'${model}'-text-only/tashkeela+clartts'

    # text + asr
    python train_lightning.py --config configs/${model}.yml --opts \
        DATA.TRAIN_PATH 'data/clartts/train_no_special.txt' \
        MODEL.USE_ASR True \
        TRAIN.SAVE_DIR 'results/'${model}'-text+asr/clartts'

    python train_lightning.py --config configs/${model}.yml --opts \
        DATA.TRAIN_PATH 'data/clartts/train_no_special.txt' \
        MODEL.USE_ASR True \
        MODEL.PRETRAINED_PATH 'results/'${model}'-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt' \
        MODEL.LOAD_TEXT_BRANCH_ONLY True \
        TRAIN.SAVE_DIR 'results/'${model}'-text+asr/tashkeela+clartts'
done