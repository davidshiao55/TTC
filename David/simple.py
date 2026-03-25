from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    outputs = llm.generate(prompts, sampling_params)
    print()
    print()
    print()
    print(outputs[0].outputs)
if __name__ == "__main__":
    main()