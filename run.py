"""
A TEI XML Named Entity Recognition (NER) processor that uses LLaMA 3.1 to identify and
annotate named entities while preserving document integrity. The processor follows a
multi-stage pipeline:

Process Pipeline:
    Input XML                Output XML
        |                        ↑
        |                        |
        ↓                        |
    Load Document               Save
        |                        ↑
        |                        |
        ↓                    Validation
    LLaMA Model ----→ NER ----→ Phase
                   generation    |
                                 |
                          [Checks for]
                          - XML Structure
                          - Tag Balance
                          - Content Integrity

Usage:
    python run.py -i input.xml -o output.xml [-t] [-p] [-v] [-f]

Options:
    -i, --input        Input TEI XML file path
    -o, --output       Output annotated file path
    -t, --temperature  Temperature for sampling (0.01-1.0, lower = more rigid)
    -p, --force        Top-p sampling parameter (0.0-1.0, lower = more selective)
    -v, --verbose      Print processing details
    -f, --force        Save output even if validation fails
"""

import argparse
import difflib
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree.ElementTree import ParseError

from vllm import LLM, SamplingParams

SYSTEM_PROMPT = """You are a named entity recognition (NER) assistant designed to annotate TEI XML files. Your task is to identify and annotate named entities (persons, organizations, and locations) using the <name> element without specifying the type attribute. Follow these instructions:

1. Identify all named entities (person names, organization names, and location names) within the provided TEI content.
2. Wrap each identified entity in a <name> element.
3. Maintain the structure and integrity of the TEI document. Only modify the text by adding the <name> elements around identified entities.
4. Do not alter any other part of the TEI content, including tags, attributes, or comments.
5. Ensure the output is well-formed XML, adhering to the TEI XML schema.

Examples:
1. Input: "John Doe, a researcher at Digital Humanities Quarterly, presented his findings."
   Output: "<name>John Doe</name>, a researcher at <name>Digital Humanities Quarterly</name>, presented his findings."

2. Input: "The conference was held in New York and featured a panel on digital scholarship."
   Output: "The conference was held in <name>New York</name> and featured a panel on digital scholarship."

3. Input: "In a recent issue, Jane Smith reviewed the book 'Digital Textual Studies.'"
   Output: "In a recent issue, <name>Jane Smith</name> reviewed the book 'Digital Textual Studies.'"

Your output should be the entire annotated TEI document, preserving its original structure."""


class ValidationResult:
    def __init__(self):
        self.passed = True
        self.messages = []
        self.statistics = {}

    def add_message(self, message: str, is_error: bool = False):
        self.messages.append(message)
        if is_error:
            self.passed = False

    def add_statistic(self, key: str, value: any):
        self.statistics[key] = value


def create_conversation(tei_file: str) -> List[dict]:
    """Create the conversation format required by the LLM."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Please annotate named entities in the following text: {tei_file}",
        },
    ]


def mask_ner_tags(text: str) -> str:
    """Replace <name> tags with placeholders for comparison."""
    return re.sub(r"</?name>", "", text)


def check_xml_validity(text: str) -> Tuple[bool, str]:
    """Validate XML structure."""
    try:
        ET.fromstring(text)
        return True, "XML is well-formed"
    except ParseError as e:
        return False, f"XML validation failed: {str(e)}"


def count_ner_tags(text: str) -> Dict[str, int]:
    """Count NER tags and their distribution."""
    opening_tags = len(re.findall(r"<name>", text))
    closing_tags = len(re.findall(r"</name>", text))
    return {"opening_tags": opening_tags, "closing_tags": closing_tags}


def validate_changes(original: str, generated: str) -> ValidationResult:
    """
    Comprehensive validation of the generated TEI document.

    Checks:
    1. Content integrity (only NER tags added)
    2. XML validity
    3. Tag balance
    4. Entity distribution
    """
    result = ValidationResult()

    # check XML validity
    is_valid_xml, xml_message = check_xml_validity(generated)
    if not is_valid_xml:
        result.add_message(xml_message, is_error=True)
    else:
        result.add_message("✓ XML validation passed")

    # check tag counts
    tag_counts = count_ner_tags(generated)
    if tag_counts["opening_tags"] != tag_counts["closing_tags"]:
        result.add_message(
            f"✗ Unbalanced NER tags: {tag_counts['opening_tags']} opening vs {tag_counts['closing_tags']} closing",
            is_error=True,
        )
    else:
        result.add_message(f"✓ Found {tag_counts['opening_tags']} balanced NER tags")

    # check content integrity
    masked_generated = mask_ner_tags(generated)
    if original == masked_generated:
        result.add_message("✓ Content integrity check passed")
    else:
        result.add_message("✗ Unexpected modifications found:", is_error=True)
        differ = difflib.Differ()
        diff = list(
            differ.compare(original.splitlines(), masked_generated.splitlines())
        )
        differences = [line for line in diff if line.startswith(("+ ", "- ", "? "))]
        for diff in differences[:5]:  # show only first 5 differences
            result.add_message(f"  {diff}")
        if len(differences) > 5:
            result.add_message(f"  ... and {len(differences) - 5} more differences")

    # add statistics
    result.add_statistic("ner_tags", tag_counts["opening_tags"])
    result.add_statistic("xml_valid", is_valid_xml)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Process TEI XML files with NER annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input TEI XML file path")
    parser.add_argument(
        "-o", "--output", required=True, help="Output annotated file path"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print processing details"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Save output even if validation fails",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling (0.01-1.0, lower = more rigid)"
    )
    parser.add_argument(
        "-p",
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter (0.0-1.0, lower = more selective)"
    )
    args = parser.parse_args()

    # sanity check
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Processing {args.input}...")

    # initialize model
    try:
        hf_model = "meta-llama/Llama-3.1-8B-Instruct"
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=1024 * 15
        )

        llm = LLM(model=hf_model, device="cuda", tensor_parallel_size=1)
    except Exception as e:
        print(f"Error initializing model: {str(e)}", file=sys.stderr)
        return 1

    # read input file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            original_text = f.read()
    except Exception as e:
        print(f"Error reading input file: {str(e)}", file=sys.stderr)
        return 1

    # generate annotations
    print("Generating annotations...")
    conversation = create_conversation(original_text)
    try:
        output = llm.chat(
            messages=conversation,
            sampling_params=sampling_params,
            use_tqdm=args.verbose,
        )
        generated_text = output[0].outputs[0].text
    except Exception as e:
        print(f"Error during generation: {str(e)}", file=sys.stderr)
        return 1

    # validation
    print("\nValidating output...")
    validation = validate_changes(original_text, generated_text)

    # print validation results
    print("\nValidation Results:")
    for message in validation.messages:
        print(message)

    if args.verbose:
        print("\nStatistics:")
        for key, value in validation.statistics.items():
            print(f"- {key}: {value}")

    # save output if validation passed or force flag is set
    if validation.passed or args.force:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(generated_text)
            print(f"\nOutput saved to: {args.output}")

            # save raw output for debugging
            pickle_path = Path(args.output).with_suffix(".pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(output, f)
            if args.verbose:
                print(f"Raw output saved to: {pickle_path}")
        except Exception as e:
            print(f"Error saving output: {str(e)}", file=sys.stderr)
            return 1
    else:
        print(
            "\nValidation failed. Use --force to save output anyway.", file=sys.stderr
        )
        return 1

    return 0 if validation.passed else 1


if __name__ == "__main__":
    exit(main())
