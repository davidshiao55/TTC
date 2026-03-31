from importlib import metadata
from vllm import ModelRegistry
from vllm import LLM
import pathlib, vllm_add_dummy_model

# llm = LLM(model="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B", enforce_eager=True)

# check available plugins
eps = metadata.entry_points(group="vllm.general_plugins")
print(f"Available General Plugins: {eps}")

# check plugin location
print("module path:", pathlib.Path(vllm_add_dummy_model.__file__).parent)
print("dist info  :", metadata.distribution("vllm-add-dummy-model").locate_file(""))


for ep in eps:
    if ep.name == "register_dummy_model":
        try:
            func = ep.load()          # 真正 import
            print(f"{ep.name} -> OK")
        except Exception as err:
            print(f"{ep.name} -> 导入失败:", type(err).__name__, err)


