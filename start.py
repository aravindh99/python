from neuttsair.neutts import NeuTTSAir
import soundfile as sf

with NeuTTSAir( backbone_repo="neuphonic/neutts-air-q4-gguf", backbone_device="cpu", codec_repo="neuphonic/neucodec", codec_device="cpu") as tts:
    ref_text_path = "samples/dave.txt"
    ref_audio_path = "samples/dave.wav"

    ref_text = open(ref_text_path, "r").read().strip()
    ref_codes = tts.encode_reference(ref_audio_path)

    print("Type text to synthesize (blank to exit)...")
    idx = 1
    while True:
        try:
            input_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not input_text:
            break

        wav = tts.infer(input_text, ref_codes, ref_text)
        out_path = f"out_{idx}.wav"
        sf.write(out_path, wav, 24000)
        print(f"Saved {out_path}")
        idx += 1
