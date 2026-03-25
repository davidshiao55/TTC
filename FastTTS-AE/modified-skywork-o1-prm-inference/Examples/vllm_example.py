from vllm import LLM
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
from model_utils.io_utils import prepare_input, sigmoid

prm_model_path = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(prm_model_path, trust_remote_code=True)
model = LLM(model=prm_model_path, enforce_eager=True)
step_token = "\n"

datas = [
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To determine how much money Janet makes every day at the farmers' market, we need to follow these steps:\n1. **Calculate the total number of eggs laid by the ducks per day.**\n   Janet's ducks lay 16 eggs per day.\n2. **Determine the number of eggs Janet uses each day.**\n   - She eats 3 eggs for breakfast every morning.\n   - She bakes muffins for her friends every day with 4 eggs.\n   So, the total number of eggs used per day is:\n   \\[\n   3 + 4 = 7 \\text{ eggs}\n   \\]\n3. **Calculate the number of eggs Janet sells at the farmers' market each day.**\n   Subtract the number of eggs used from the total number of eggs laid:\n   \\[\n   16 - 7 = 9 \\text{ eggs}\n   \\]\n4. **Determine how much money Janet makes from selling the eggs.**\n   She sells each egg for $2, so the total amount of money she makes is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]\nTherefore, the amount of money Janet makes every day at the farmers' market is \\(\\boxed{18}\\)."
    },
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To find out how much Janet makes every day at the farmers' market, we need to follow these steps:\n1. Calculate the total number of eggs laid by the ducks: \n   - Janet's ducks lay 16 eggs per day.\n2. Calculate the total number of eggs consumed by Janet and used for baking:\n   - Janet eats 3 eggs for breakfast.\n   - She bakes 4 eggs for muffins.\n   - Total eggs used: 3 (breakfast) + 4 (baking) = 7 eggs.\n3. Calculate the remaining number of eggs for sale:\n   - Total eggs laid: 16\n   - Eggs used: 7\n   - Remaining eggs: 16 - 7 = 9 eggs\n4. Calculate the total amount of money made at the farmers' market:\n   - Price per egg: $2\n   - Number of eggs sold: 9\n   - Total money made: 9 * $2 = $18\nTherefore, Janet makes $\\boxed{18}$ dollars every day at the farmers' market."
    }
]

simple_datas = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is Paris.\nThis is a beautiful city.",
    "The future of AI is",
]

# 

def test_vllm(task):
    # input preprocessing
    if task == "simple":
        processed_data = []
        for d in simple_datas:
            response_ids = []
            steps = []
            reward_flags = []
            step_token_id = tokenizer.encode(step_token)[-1]
            for idx, step in enumerate(d.split(step_token)):
                if step != "":
                    step_ids = tokenizer.encode(step)
                else:
                    step_ids = []
                step_ids += [step_token_id]
                step = step + step_token
                flag = [0] * len(step_ids)
                flag[-1] = 1
                response_ids.extend(step_ids)
                reward_flags.extend(flag)
                steps.append(step)
            input_ids = response_ids
            processed_data.append((input_ids, steps, reward_flags))
        input_ids, steps, reward_flags = zip(*processed_data)
        # check
        print(f"input_ids:\n {input_ids}")
        print(f"steps:\n {steps}")
        print(f"reward_flags:\n {reward_flags}")
    elif task == "conversation":
        processed_data = [prepare_input(d["problem"], d["response"], tokenizer=tokenizer, step_token="\n") for d in datas]
        input_ids, steps, reward_flags = zip(*processed_data)
        # check
        print(f"input_ids:\n {input_ids}")
        print(f"steps:\n {steps}")
        print(f"reward_flags:\n {reward_flags}")
    else: 
        raise ValueError(f"Invalid task: {task}")

    # model inference
    prompts = [TokensPrompt(prompt_token_ids=input_ids) for input_ids in input_ids]
    rewards = model.encode(prompts)
    # step rewards output processing
    step_rewards = []
    for reward, reward_flag in zip(rewards, reward_flags):
        step_reward = []
        reward_embedding = reward.outputs.data.tolist()
        print(f"reward_embedding: {reward_embedding}")
        print(f"reward_flag: {reward_flag}")
        for idx, flag in enumerate(reward_flag):
            if flag == 1:
                step_reward.append(sigmoid(reward_embedding[idx][0]))
        step_rewards.append(step_reward)

    print(f"step_rewards:\n {step_rewards}")
    print(f"Type of step_rewards: {type(step_rewards)}")
    print(f"Type of step_rewards[0]: {type(step_rewards[0])}")
    print(f"Type of step_rewards[0][0]: {type(step_rewards[0][0])}")


if __name__ == "__main__":
    # test_vllm("simple")
    test_vllm("conversation")