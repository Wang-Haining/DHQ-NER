from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    """You are a named entity recognition (NER) assistant designed to annotate TEI XML files. Your task is to identify and annotate named entities (persons, organizations, and locations) using the <name> element without specifying the type attribute. Follow these instructions:

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

Your output should be the entire annotated TEI document, preserving its original structure.
"""
)


def create_conversation(tei_file):
    # generate a conversation template that includes the drug name
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                f"Please annotate named entities in the following text: {tei_file}"
            )
        }
    ]

hf_model = 'meta-llama/Llama-3.1-8B-Instruct'
sampling_params = SamplingParams(temperature=0.6, top_p=0.95,
                                 max_tokens=1024*10)

llm = LLM(model=hf_model,
          tensor_parallel_size=1,
          # dtype='bf16'
          )


tei_file = open('000760_without_names.xml', 'r').read()
conversation = create_conversation(tei_file)

output = llm.chat(
    messages=conversation,
    sampling_params=sampling_params,
    use_tqdm=True
)

print(output)

with open('annotated_output.xml', 'w') as f:
    f.write(output.outputs[0].text)

print("Annotated TEI file saved as 'annotated_output.xml'.")