"""LTX-2.5 DistilledPipeline с моделями, загруженными один раз, — генерирует список шотов подряд.
cd /root/LTX-2 && uv run python /root/work/ltx_batch.py <shots.json> <out_dir>
shots.json: {"shots":[{"id":..., "seconds": int, "prompt": "...", "seed": int}]}. Готовые файлы пропускаются.
"""
import json, logging, os, sys, time
import torch
import ltx_pipelines.distilled as D

M = "/root/LTX-2/models/ltx-2.5"
QUANT = os.environ.get("LTX_QUANT", "fp8-cast")
TRANSFORMER = os.environ.get("LTX_TRANSFORMER", "ltx-2.5-22b-distilled-transformer-bf16.safetensors")
COMPILE = ["--compile"] if os.environ.get("LTX_COMPILE") == "1" else []
BASE = ["--transformer-path", f"{M}/diffusion_models/{TRANSFORMER}",
        "--text-encoder-path", f"{M}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "--video-vae-path", f"{M}/vae/ltx-2.5-video-vae-bf16.safetensors",
        "--audio-vae-path", f"{M}/vae/ltx-2.5-audio-vae-bf16.safetensors",
        "--spatial-upsampler-path", f"{M}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "--prompt", "x", "--output-path", "/tmp/x.mp4", "--width", "1280", "--height", "704", "--num-frames", "121",
        "--frame-rate", "24", "--quantization", QUANT] + COMPILE


@torch.inference_mode()
def main():
    logging.basicConfig(level=logging.WARNING)
    cfg, out = json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    sys.argv = [sys.argv[0]] + BASE  # resolve_cli_params читает sys.argv
    params = D.resolve_cli_params(distilled=True)
    parser = D.add_chunk_layout_args(D.add_keyframe_decode_arg(D.add_generated_keyframes_arg(
        D.default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True))))
    args = parser.parse_args(BASE)  # noqa
    hdr = D.resolve_hdr_color_space(images=args.images, hdr=args.hdr)
    vm, am = D.markers_from_args(args, device=D.get_device(), color_space=hdr)
    t0 = time.time()
    pipe = D.DistilledPipeline(model_paths=args.model_paths, spatial_upsampler_path=args.spatial_upsampler_path, loras=(),
                               quantization=args.quantization, compilation_config=args.compile, offload_mode=args.offload_mode,
                               prompt_enhancer_gemma_root=None, diffvae_optimization=args.diffvae_optimization)
    print(f"LOADED {time.time() - t0:.0f}s", flush=True)
    for s in cfg["shots"]:
        dst = f"{out}/{s['id']}.mp4"
        if os.path.exists(dst):
            continue
        t0 = time.time()
        try:
            chunks, _, _ = pipe.stream_chunks(prompt=s["prompt"], seed=s.get("seed", 42), height=704, width=1280,
                                              num_frames=int(s["seconds"]) * 24 + 1, frame_rate=24, images=args.images,
                                              vae_dtype=D.vae_dtype_for_hdr(hdr, torch.bfloat16), color_space=hdr,
                                              enhance_prompt=False, enhance_static_cache=False, tiling_config=D.AUTO_TILING,
                                              generated_keyframes=0, decode_with_keyframes=False,
                                              chunk_config=D.chunk_config_from_args(args))
            D.encode_media_segments(chunks, fps=24, output_path=dst + ".tmp.mp4", video_chunks_number=None,
                                    color_space=hdr, video_marker=vm, audio_marker=am)
            os.replace(dst + ".tmp.mp4", dst)
            print(f"SHOT {s['id']} ok {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"SHOT {s['id']} FAIL {e}", flush=True)
            torch.cuda.empty_cache()
    print("LTX_DONE", flush=True)


if __name__ == "__main__":
    main()
