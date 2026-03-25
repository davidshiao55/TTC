from vllm import LLM
from vllm.model_executor.models.qwen2_rm import Qwen2ForProcessRewardModel

# prompt = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nSue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?<|im_end|>\n<|im_start|>assistant\nTo find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.<extra_0>On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.<extra_0>On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.<extra_0>To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30}).<extra_0><|im_end|><|endoftext|>"

prompts = [
    "Hello, my name is",
    "Hello, my name is"
]

llm = LLM(model="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B", task="reward")
outputs = llm.encode(prompts)

print("Number of outputs:", len(outputs))
for i, output in enumerate(outputs):
    print(f"Output {i}:", output.outputs.data.tolist())