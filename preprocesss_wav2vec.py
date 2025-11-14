from datasets import (
    DatasetDict,
    Dataset,
    load_dataset,
    concatenate_datasets,
    load_from_disk,
)
from tqdm import tqdm
import numpy as np
import soundfile as sf
import tempfile
import torchaudio
from contextlib import contextmanager
import os
import sys
import torch
import resource
import multiprocessing as mp
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# ---------- Utility Functions ----------


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def resample_audio_to_16k(audio_data, sampling_rate):
    """Convert audio to mono and resample to 16kHz."""
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)  # stereo → mono
    tensor_audio = torchaudio.functional.resample(
        torch.tensor(audio_data), orig_freq=sampling_rate, new_freq=16000
    )
    return tensor_audio.numpy()


def extract_emotion2vec_features(model, audio_path):
    with suppress_output():
        result = model.generate(
            audio_path, output_dir=None, granularity="utterance", extract_embedding=True
        )
    if isinstance(result, list):
        return result
    elif isinstance(result, dict):
        return result.get("embedding", [])
    else:
        return []


def extract_wav2vec2_features(processor, model, audio_data):
    """Extract wav2vec2 embeddings from raw audio numpy array."""
    input_values = processor(
        audio_data, sampling_rate=16000, return_tensors="pt"
    ).input_values
    with torch.no_grad():
        outputs = model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


# ---------- Core Dataset Modifier (Streaming + Chunked) ----------


def modify_dataset_with_encoder(
    dataset,
    encoder_type="emotion2vec",
    model=None,
    chunk_size=2000,
    output_dir="processed_chunks",
):
    """
    Processes dataset split-by-split in streaming mode.
    - Keeps all columns except 'audio'
    - Adds an embedding column
    - Saves intermediate chunks to disk to avoid OOM
    """

    os.makedirs(output_dir, exist_ok=True)
    chunk_idx = 0
    processed_samples = []

    for split_name, split_data in dataset.items():
        print(f"🔄 Processing split: {split_name}")

        for i, sample in enumerate(tqdm(split_data)):
            # Copy all non-audio fields
            sample_out = {k: v for k, v in sample.items() if k != "audio"}

            audio = sample["audio"]
            audio_data = audio["array"]
            sampling_rate = audio["sampling_rate"]

            # Resample
            resampled_audio = resample_audio_to_16k(audio_data, sampling_rate)

            # Extract embedding
            if encoder_type == "emotion2vec":
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, resampled_audio, 16000)
                    tmp_path = tmp.name
                try:
                    embedding = extract_emotion2vec_features(model, tmp_path)
                except Exception as e:
                    print(f"⚠️ Error processing sample {i}: {e}")
                    embedding = []
                os.remove(tmp_path)

            elif encoder_type == "wav2vec2":
                processor, wav2vec_model = model
                try:
                    embedding = extract_wav2vec2_features(
                        processor, wav2vec_model, resampled_audio
                    )
                except Exception as e:
                    print(f"⚠️ Error processing sample {i}: {e}")
                    embedding = []
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")

            # Add embedding column
            sample_out[f"{encoder_type}_features"] = embedding
            processed_samples.append(sample_out)

            # Periodically flush to disk to free memory
            if len(processed_samples) >= chunk_size:
                chunk_ds = Dataset.from_list(processed_samples)
                chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx}")
                chunk_ds.save_to_disk(chunk_path)
                print(f"💾 Saved {len(processed_samples)} samples to {chunk_path}")
                processed_samples = []
                chunk_idx += 1

        # Final leftover chunk
        if processed_samples:
            chunk_ds = Dataset.from_list(processed_samples)
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx}")
            chunk_ds.save_to_disk(chunk_path)
            print(f"💾 Saved final {len(processed_samples)} samples to {chunk_path}")
            processed_samples = []
            chunk_idx += 1

    print("✅ All chunks saved to disk!")
    return output_dir


def merge_chunks_and_push(output_dir, repo_id):
    """Loads all chunks from disk, merges them, and pushes to the Hub."""
    print("📦 Merging chunks...")
    chunk_paths = sorted(
        [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("chunk_")
        ]
    )
    datasets_list = [load_from_disk(path) for path in chunk_paths]
    merged = concatenate_datasets(datasets_list)
    print(f"✅ Merged dataset with {len(merged)} samples")

    final_dict = DatasetDict({"train": merged})
    print("🚀 Pushing to Hub...")
    final_dict.push_to_hub(repo_id)
    print("🎉 Upload complete!")


# ---------- Memory + Worker Logic ----------


def limit_memory(max_mb):
    max_bytes = max_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))


def worker_main():
    limit_memory(22000)  # 22 GB cap

    try:
        dataset = load_dataset("cairocode/samsemo-audio", streaming=True)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        output_dir = modify_dataset_with_encoder(
            dataset,
            encoder_type="wav2vec2",
            model=(processor, wav2vec_model),
            chunk_size=2000,  # adjust based on RAM
            output_dir="wav2vec_chunks",
        )

        merge_chunks_and_push(output_dir, "SAMSEMO_Wav2Vec2_Text")

    except MemoryError:
        print("❌ Worker ran out of memory! Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Worker crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    p = mp.Process(target=worker_main)
    p.start()
    p.join()

    if p.exitcode != 0:
        print(f"Worker exited with code {p.exitcode}. Main process is safe!")
    else:
        print("✅ Worker finished successfully.")
