
# decode
for model in "lstm" "transformer"
do
    echo "Running inference for model: $model"

    # text only
    python inference.py --config configs/${model}.yml \
        DATA.TEST_PATH 'data/clartts/ClarTTS_test_Ref.txt' \
        MODEL.USE_ASR False \
        MODEL.PRETRAINED_PATH 'checkpoints/'${model}'-text-only/tashkeela/tensorboard/version_0/checkpoints/best_model.ckpt' \
        OUTPUT_PATH 'results/'${model}'-text-only/tashkeela/predictions.txt' \
        INFERENCE.USE_ASR False

    python inference.py --config configs/${model}.yml \
        DATA.TEST_PATH 'data/clartts/ClarTTS_test_Ref.txt' \
        MODEL.USE_ASR False \
        MODEL.PRETRAINED_PATH 'checkpoints/'${model}'-text-only/clartts/tensorboard/version_0/checkpoints/best_model.ckpt' \
        INFERENCE.OUTPUT_PATH 'results/'${model}'-text-only/clartts/predictions.txt' \
        INFERENCE.USE_ASR False

    python inference.py --config configs/${model}.yml \
        DATA.TEST_PATH 'data/clartts/ClarTTS_test_Ref.txt' \
        MODEL.USE_ASR False \
        MODEL.PRETRAINED_PATH 'checkpoints/'${model}'-text-only/tashkeela+clartts/tensorboard/version_0/checkpoints/best_model.ckpt' \
        INFERENCE.OUTPUT_PATH 'results/'${model}'-text-only/tashkeela+clartts/predictions.txt' \
        INFERENCE.USE_ASR False

    # text + asr
    python inference.py --config configs/${model}.yml \
        DATA.TEST_PATH 'data/clartts/ClarTTS_test_Ref.txt' \
        MODEL.USE_ASR True \
        MODEL.PRETRAINED_PATH 'checkpoints/'${model}'-text+asr/clartts/tensorboard/version_0/checkpoints/best_model.ckpt' \
        INFERENCE.OUTPUT_PATH 'results/'${model}'-text+asr/clartts/predictions.txt'
    
    python inference.py --config configs/${model}.yml \
        DATA.TEST_PATH 'data/clartts/ClarTTS_test_Ref.txt' \
        MODEL.USE_ASR True \
        MODEL.PRETRAINED_PATH 'checkpoints/'${model}'-text+asr/tashkeela+clartts/tensorboard/version_0/checkpoints/best_model.ckpt' \
        INFERENCE.OUTPUT_PATH 'results/'${model}'-text+asr/tashkeela+clartts/predictions.txt'
done

# evaluate
for model in "lstm" "transformer"
do
    python eval.py -ofp results/${model}-text-only/tashkeela/predictions.txt -tfp data/clartts/ClarTTS_test_Ref.txt
    python eval.py -ofp results/${model}-text-only/clartts/predictions.txt -tfp data/clartts/ClarTTS_test_Ref.txt
    python eval.py -ofp results/${model}-text-only/tashkeela+clartts/predictions.txt -tfp data/clartts/ClarTTS_test_Ref.txt
    python eval.py -ofp results/${model}-text+asr/clartts/predictions.txt -tfp data/clartts/ClarTTS_test_Ref.txt
    python eval.py -ofp results/${model}-text+asr/tashkeela+clartts/predictions.txt -tfp data/clartts/ClarTTS_test_Ref.txt
done