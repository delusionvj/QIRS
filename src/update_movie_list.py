"""
Populate config.yaml's `movie_list` from the movie_titles.json that
src/prepare_movielens.py generates, instead of copy-pasting titles by hand.

Usage (after running prepare_movielens.py):
    python src/update_movie_list.py --max-movies 50

Optionally, clean up each title with a small local Qwen model (run in-process
via Hugging Face `transformers`, no separate server needed — this is what
lets it run on a rented GPU box like vast.ai without extra setup) before
writing it. Fixes things like MovieLens' "Truth About Cats & Dogs, The"
trailing-article style, stray punctuation/whitespace, etc.:
    pip install transformers accelerate
    python src/update_movie_list.py --max-movies 50 --clean-titles

The first run downloads the model from Hugging Face (~9GB for the default
Qwen/Qwen3.5-4B) and caches it locally; subsequent runs reuse the cache.

This edits config.yaml in place, replacing only the `movie_list: ...` line/
block and leaving every other line (including comments) untouched.
"""

import argparse
import json

import yaml

DEFAULT_HF_MODEL = "Qwen/Qwen3.5-4B"

_MODEL_CACHE = {}


def _load_model(model_name: str, device: str):
    """Load (and cache) the tokenizer/model so it's only loaded into memory once
    per process, not once per title — loading a multi-GB checkpoint per call
    would make this unusably slow."""
    key = (model_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} onto {device} (first run downloads it from "
          f"Hugging Face and can take a while) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model.to("cpu")
    model.eval()

    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def clean_title_with_hf(title: str, model_name: str, device: str, max_new_tokens: int = 32) -> str:
    """
    Ask a local Qwen model (via Hugging Face transformers) to clean up a
    single movie title. Falls back to the original, unmodified title on any
    failure (load error, generation error, empty/garbage response, etc.)
    rather than raising — a single bad title should never abort the whole batch.
    """
    prompt = (
        "Clean up this movie title for a movie database lookup. Fix formatting "
        "issues such as a trailing article moved to the end (e.g. \"Truth About "
        "Cats & Dogs, The\" -> \"The Truth About Cats & Dogs\"), stray punctuation, "
        "or extra whitespace. Do not translate it, add a year, or change which "
        "movie it refers to. Respond with ONLY the cleaned title on a single "
        "line - no quotes, no explanation.\n\n"
        f"Title: {title}"
    )

    try:
        import torch

        tokenizer, model = _load_model(model_name, device)

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_ids.shape[-1]:]
        raw = tokenizer.decode(generated, skip_special_tokens=True)
    except Exception as e:
        print(f"  [warn] Local generation failed for '{title}': {e}. Keeping original title.")
        return title

    # Small local models sometimes ignore "single line, no quotes" - defend
    # against that rather than trusting the output blindly.
    cleaned = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    cleaned = cleaned.strip('"').strip("'").strip()

    if not cleaned or len(cleaned) > len(title) + 40:
        print(f"  [warn] Suspicious cleaned output for '{title}': {raw!r}. Keeping original title.")
        return title

    return cleaned


def clean_titles(titles: list, model_name: str, device: str) -> list:
    """Clean each title via the local Qwen model, preserving order and
    de-duplicating (cleaning can collapse two raw titles into the same string)."""
    cleaned = []
    seen = set()
    for i, title in enumerate(titles, 1):
        print(f"  Cleaning {i}/{len(titles)}: {title!r} ...")
        new_title = clean_title_with_hf(title, model_name, device)
        if new_title != title:
            print(f"    -> {new_title!r}")
        if new_title not in seen:
            seen.add(new_title)
            cleaned.append(new_title)
    return cleaned


def build_movie_list_lines(titles: list) -> list:
    """Render titles as YAML block-list lines, matching config.yaml's indentation style."""
    lines = ["movie_list:"]
    for title in titles:
        # yaml.dump on a single-item list gives us a correctly quoted/escaped
        # scalar (handles embedded quotes, colons, etc.) without pulling in a
        # second templating mechanism.
        item = yaml.dump([title], default_flow_style=True, allow_unicode=True).strip()
        # item looks like '[Some Title]' or ["Some: Title"] - strip the brackets
        inner = item[1:-1]
        lines.append(f"  - {inner}")
    return lines


def replace_movie_list_block(config_text: str, new_lines: list) -> str:
    """
    Replace the movie_list entry (whether it's a one-line placeholder like
    `movie_list: []` or a previously expanded block with indented `- ...`
    items) with new_lines, leaving every other line untouched.
    """
    lines = config_text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if line.startswith("movie_list:"):
            start = i
            break

    if start is None:
        raise SystemExit("Could not find a 'movie_list:' entry in the config to replace")

    # The block extends over every subsequent line that's an indented list
    # item (starts with whitespace then "-"). The first line that ISN'T
    # (a blank line, a comment, or the next top-level key) ends the block.
    end = start + 1
    while end < len(lines) and lines[end][:1] in (" ", "\t") and lines[end].lstrip().startswith("-"):
        end += 1

    return "\n".join(lines[:start] + new_lines + lines[end:]) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Populate config.yaml's movie_list from movie_titles.json"
    )
    parser.add_argument("--titles-file", type=str, default="data/Input/movie_titles.json",
                         help="Path to the movie_titles.json written by prepare_movielens.py")
    parser.add_argument("--config", type=str, default="config.yaml",
                         help="Path to config.yaml to update in place")
    parser.add_argument("--max-movies", type=int, default=50,
                         help="Cap on how many titles to insert (each triggers an OpenAI API "
                              "call downstream, so keep this small for a first test run)")
    parser.add_argument("--clean-titles", action="store_true",
                         help="Clean each title with a local Qwen model (via Hugging Face "
                              "transformers, run in-process) before writing it")
    parser.add_argument("--hf-model", type=str, default=DEFAULT_HF_MODEL,
                         help=f"Hugging Face model repo to use for cleaning (default: {DEFAULT_HF_MODEL})")
    parser.add_argument("--device", type=str, default=None,
                         help="Device to run the model on: 'cuda', 'cpu', etc. "
                              "(default: auto-detects CUDA, falls back to CPU)")
    args = parser.parse_args()

    with open(args.titles_file, "r", encoding="utf-8") as f:
        titles = json.load(f)

    if args.max_movies is not None:
        titles = titles[:args.max_movies]

    if not titles:
        raise SystemExit(f"No titles found in {args.titles_file}")

    if args.clean_titles:
        if args.device:
            device = args.device
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cleaning {len(titles)} titles with {args.hf_model} on {device} ...")
        titles = clean_titles(titles, args.hf_model, device)

    with open(args.config, "r", encoding="utf-8") as f:
        config_text = f.read()

    new_lines = build_movie_list_lines(titles)
    new_config_text = replace_movie_list_block(config_text, new_lines)

    with open(args.config, "w", encoding="utf-8") as f:
        f.write(new_config_text)

    print(f"Wrote {len(titles)} titles into {args.config}'s movie_list "
          f"(from {len(titles)} of the entries in {args.titles_file}).")


if __name__ == "__main__":
    main()
