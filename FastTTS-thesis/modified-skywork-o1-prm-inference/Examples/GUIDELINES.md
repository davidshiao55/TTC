# Guidelines

## Install the plugin
- This implementation uses `vllm==0.9.2`
- Use `pip install -e .` to install the plugin (will automatically do model registration)
- To check if the plugin is installed, use `confirm_plugin.py`
- If you want to reinstall the plugin, delete the `.egg-info` folder, also make sure there's no installed plugin in system path. 

## Run reward task
- The model will not automatically derive step rewards (as the setup they provided in transformers), it is more like a RM (e.g. `Qwen/Qwen2.5-Math-RM-72B`)
- Preprocessing and postprocessing code are provided in the scripts
- Try `vllm_example.py` for vllm version, you may verify the results using `transformers_example.py`