#!/bin/bash
# Скачивание весов LTX-2.5 в /root/LTX-2/models/ltx-2.5. Пока идёт — на машине лежит /root/hold (сторож не ставит на паузу).
set -x
touch /root/hold
M=/root/LTX-2/models/ltx-2.5
FILES="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
vae/ltx-2.5-video-vae-bf16.safetensors
vae/ltx-2.5-audio-vae-bf16.safetensors
latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
for v in ${LTX_VARIANTS:-bf16}; do
  case $v in
    bf16)  FILES="$FILES
diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" ;;
    nvfp4) FILES="$FILES
diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors" ;;
  esac
done
T0=$(date +%s)
hf download Lightricks/LTX-2.5 $FILES --local-dir $M
du -sh $M/*/
echo "STAGE weights_done $(( $(date +%s) - T0 ))s $(date)" >> /root/logs/setup.log
rm -f /root/hold
