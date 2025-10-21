import base64
import glob
import json
import logging
import os
import re
import sys
from argparse import ArgumentParser, Namespace
from io import BytesIO
from pathlib import Path

from numpy import array_split
from PIL import Image
from tqdm.auto import tqdm

# --- add project root to sys.path when run as a script ---
ROOT = Path(__file__).resolve().parents[1]  # .../open-maia
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------

from maia_api import Synthetic_System, System
from utils.agents.agent import BaseAgent as Agent
from utils.agents.factory import create_agent
from utils.flux import FluxDev


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Agent Neuron Description Evaluation")
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model on (e.g., 'cuda:0' or 'cpu').",
        default="cpu",
    )  # Default to CPU if not specified
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to the JSON file containing neuron labels.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent to use for generating prompts. Options: 'gpt-4o', 'claude', 'local-google/gemma3-27b-it'.",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        help="Base URL for the local agent server, if applicable.",
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of prompts to generate", default=7
    )

    parser.add_argument(
        "--chunk_id", type=int, required=True, help="Chunk to process", default=0
    )

    parser.add_argument(
        "--total_chunks", type=int, required=True, help="Total chunks", default=1
    )
    return parser.parse_args()


def setup_logging():
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_prompts(agent: Agent, instruction: str) -> list[str]:
    history = [
        {"role": "user", "content": [{"type": "text", "text": instruction}]},
    ]
    response = agent.ask(history)
    return [line for line in response.splitlines() if line.strip()]


def generate_images(text2image: FluxDev, prompts: list[str]) -> list[Image]:
    images = text2image.generate_batch(prompts, max_batch=15)
    return images


def image2str(image: Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode("ascii")
    return image


if __name__ == "__main__":
    # Instruction Prompts to generate positive and negative prompts
    POS_INSTRUCTIONS: str = (
        "You will receive an input label describing the concept(s) detected by a neuron within a deep neural network for computer vision. "
        "Based on that label, write a list of {n} prompts for a text-to-image model. These prompts will be used to generate images that maximally activate the neuron. "
        "The prompts should exclusively describe the image content. If multiple concepts are described (e.g. separated by 'or') they should be represented by different prompts. "
        "Each prompt should be on a new line, do not number the list. {label}"
    )
    NEG_INSTRUCTIONS: str = (
        "You will receive an input label describing the concept(s) detected by a neuron within a deep neural network for computer vision. "
        "Based on that label, write a list of {n} prompts for a text-to-image model. These prompts will be used to generate images that minimally activate the neuron, because they contain content that is unrelated to the neuron label. "
        "Do not include any content or words related to the neuron label. The prompts should exclusively describe the image content. If multiple concepts are described (e.g. separated by 'or') they should be represented by different prompts. "
        "Each prompt should be on a new line, do not number the list. {label}"
    )

    # Setup logging
    setup_logging()

    # Argument parser
    args = parse_args()
    logging.info("Arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    # Load neuron labels
    label_paths = sorted(glob.glob(f"{args.labels}/*/*/*/history.json"))

    # Filter already computed
    label_paths = [
        p
        for p in label_paths
        if not os.path.exists(p.replace("history.json", "eval_results.json"))
    ]

    if args.chunk_id > 0 and args.total_chunks > 1:
        label_paths = list(
            array_split(label_paths, args.total_chunks)[args.chunk_id - 1]
        )
    logging.info(f"Found {len(label_paths)} neuron labels.")
    error = 0
    errors = []

    # Initialize image generator
    text2image = FluxDev(device=args.device)

    # Iterate over neuron labels
    for path in tqdm(label_paths, desc="Processing Neuron Labels"):
        model, layer, unit = path.split("/")[-4:-1]
        logging.info(f"Processing {model}/{layer}/{unit}")

        # Read history file
        with open(path) as f:
            neuron_data: str = json.load(f)[-1]["content"][0]["text"]

        # Extract Neuron Description and Labels
        desc_match = re.search(r"\[DESCRIPTION\]:\s*(.*?)\n(?=\[)", neuron_data, re.S)
        neuron_description = desc_match.group(1).strip() if desc_match else None
        neuron_labels = re.findall(r"\[LABEL\s*\d+\]:\s*(.+)", neuron_data)

        # Skip if no labels or description found
        if not neuron_labels or not neuron_description:
            logging.warning(
                f"No labels or description found for {model}/{layer}/{unit}. Skipping."
            )
            error += 1
            errors.append(f"{model}/{layer}/{unit}")
            continue

        # Create agent
        agent: Agent = create_agent(model=args.agent, base_url=args.base_url)

        # Ask for positive/negative prompts
        logging.info("Generating Positive Prompts")
        positive_prompts = get_prompts(
            agent=agent,
            instruction=POS_INSTRUCTIONS.format(
                n=args.n, label="\n".join(neuron_labels)
            ),
        )
        logging.info("Generating Negative Prompts")
        negative_prompts = get_prompts(
            agent=agent,
            instruction=NEG_INSTRUCTIONS.format(
                n=args.n, label="\n".join(neuron_labels)
            ),
        )

        # Generate images based on the prompts
        logging.info("Generating Positive Images")
        positive_images = generate_images(text2image, positive_prompts)
        logging.info("Generating Negative Images")
        negative_images = generate_images(text2image, negative_prompts)

        # Load the model and create the system
        logging.info("Loading Model")
        if "synthetic_neurons" in path:
            with open(
                f"./synthetic_neurons_dataset/labels/{layer}.json"
            ) as file:  # load the synthetic neuron labels
                gt_label = json.load(file)[int(unit)]["label"].rsplit("_")
                system = Synthetic_System(
                    int(unit), gt_label, layer, args.device.replace("cuda:", "")
                )
            transform = image2str
        else:
            system = System(
                int(unit), layer, model, args.device.replace("cuda:", ""), None
            )
            transform = image2str

        # Score the images
        logging.info("Scoring Positive Images")
        pos_scores = system.call_neuron(list(map(transform, positive_images)))[0]
        logging.info(f"Average Positive Score: {sum(pos_scores) / len(pos_scores)}")

        logging.info("Scoring Negative Images")
        neg_scores = system.call_neuron(list(map(transform, negative_images)))[0]
        logging.info(f"Average Negative Score: {sum(neg_scores) / len(neg_scores)}")

        # Save Results
        logging.info("Saving Results")
        results = {
            "average_positive_score": sum(pos_scores) / len(pos_scores),
            "average_negative_score": sum(neg_scores) / len(neg_scores),
            "positive_prompts": positive_prompts,
            "negative_prompts": negative_prompts,
            "positive_scores": pos_scores,
            "negative_scores": neg_scores,
        }

        # Eval JSON
        with open(f"{args.labels}/{model}/{layer}/{unit}/eval_results.json", "w") as f:
            json.dump(results, f, indent=4)

        # Save Positive Images
        os.makedirs(
            f"{args.labels}/{model}/{layer}/{unit}/positive_images", exist_ok=True
        )
        for i, img in enumerate(positive_images):
            img.save(
                f"{args.labels}/{model}/{layer}/{unit}/positive_images/img_{i}.png"
            )
        # Save Negative Images
        os.makedirs(
            f"{args.labels}/{model}/{layer}/{unit}/negative_images", exist_ok=True
        )
        for i, img in enumerate(negative_images):
            img.save(
                f"{args.labels}/{model}/{layer}/{unit}/negative_images/img_{i}.png"
            )

    logging.info(f"Total errors: {error}")
    logging.info(f"Errors: {errors}")
