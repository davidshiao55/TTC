#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# Force V0 since reward models are not supported in V1
os.environ["VLLM_USE_V1"] = "0"
import logging
import multiprocessing as mp
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import torch
from vllm import SamplingParams

from config import FastTTSConfig
from .tts_llm import TTSLLM

logger = logging.getLogger(__name__)


class BaseVLLMModelWrapper(ABC):
    """Base class for VLLM model wrappers."""
    
    def __init__(
        self,
        config: FastTTSConfig,
        enable_sleep_mode: bool = False,
    ):
        self.config = config
        self.enable_sleep_mode = enable_sleep_mode
            
        # Initialize model
        self.model = None
        self.process = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the VLLM model."""
        if self.model_type == "generator":
            logger.info(f"Initializing {self.model_type} model: {self.config.generator_vllm_config['model']}")
        elif self.model_type == "verifier":
            logger.info(f"Initializing {self.model_type} model: {self.config.verifier_vllm_config['model']}")

        # If sleep mode is enabled, use a separate process
        # if self.enable_sleep_mode:
        self._initialize_model_in_process()
        # else:
            # self._initialize_model_direct()
        
    def _initialize_model_direct(self):
        """Initialize model directly in current process."""
        model_kwargs = self._get_model_kwargs()
        try:
            self.model = TTSLLM(**model_kwargs)
        except Exception as e:
            raise e
        logger.info(f"{self.model_type.capitalize()} model initialized successfully")
        
    def _initialize_model_in_process(self):
        """Initialize model in a separate process to avoid sleep mode conflicts."""
        # Create a pipe for communication
        self.parent_conn, child_conn = mp.Pipe()
        
        # Start the model process
        self.process = mp.Process(
            target=self._model_process_worker,
            args=(child_conn, self._get_model_kwargs(), self.model_type)
        )
        # Set start method to 'spawn' for better CUDA isolation
        if mp.get_start_method() != 'spawn':
            logger.warning("Setting start method to 'spawn'")
            mp.set_start_method('spawn', force=True)
        self.process.start()
        
        # Wait for initialization confirmation
        result = self.parent_conn.recv()
        if result.get('error') or result.get('status') != 'initialized':
            raise RuntimeError(f"Failed to initialize {self.model_type} model: {result['error']}")
        
        logger.info(f"{self.model_type.capitalize()} model initialized successfully in separate process")
        
    def _model_process_worker(self, conn, model_kwargs, model_type):
        """Worker process for model initialization and inference."""
        try:
            import pycuda.driver as cuda

            pid = os.getpid()
            # Each process gets its own, separate CUDA context.
            current_context = cuda.Context.get_current()
            print(
                f"✅ Process PID: {pid}  |  "
                f"CUDA Context Object: {current_context}"
            )
            
            # use V0 for generator
            if model_type == 'generator':
                os.environ["VLLM_USE_V1"] = "0"
            else:
                os.environ["VLLM_USE_V1"] = "1"
            
            # Initialize the model
            model = TTSLLM(**model_kwargs)
            tokenizer = model.get_tokenizer()
            
            enable_sleep_mode = model_kwargs.get('enable_sleep_mode', False)
            if enable_sleep_mode:
                model.sleep()
            conn.send({'status': 'initialized'})
            
            # Keep the process alive and handle requests
            while True:
                try:
                    request = conn.recv()
                    if request.get('action') == 'shutdown':
                        break
                    elif request.get('action') == 'generate':
                        if enable_sleep_mode:
                            # Enhanced memory management for wake up
                            # torch.cuda.empty_cache()
                            # torch.cuda.synchronize()  # Ensure all operations complete
                            model.wake_up()
                        result = model.generate_text(
                            request['prompts'], 
                            **request.get('kwargs', {})
                        )
                        if enable_sleep_mode:
                            model.sleep()
                            # torch.cuda.empty_cache()
                            # torch.cuda.synchronize()
                        conn.send({'result': result})
                    elif request.get('action') == 'score':
                        if enable_sleep_mode:
                            # Enhanced memory management for wake up
                            # torch.cuda.empty_cache()
                            # torch.cuda.synchronize()
                            model.wake_up()
                        result = model.score_outputs(
                            request['questions'], 
                            request['outputs'], 
                            request.get('system_prompt', ''),
                            **request.get('kwargs', {})
                        )
                        if enable_sleep_mode:
                            model.sleep()
                            # torch.cuda.empty_cache()
                            # torch.cuda.synchronize()
                        conn.send({'result': result})
                    elif request.get('action') == 'get_tokenizer_info':
                        # Send basic tokenizer info
                        tokenizer_info = {
                            'name': getattr(tokenizer, 'name_or_path', 'unknown'),
                            'vocab_size': getattr(tokenizer, 'vocab_size', 0)
                        }
                        conn.send({'tokenizer_info': tokenizer_info})
                    elif request.get('action') == 'apply_chat_template':
                        result = tokenizer.apply_chat_template(
                            request['conversations'],
                            add_generation_prompt=request.get('add_generation_prompt', True),
                            continue_final_message=request.get('continue_final_message', False),
                            tokenize=request.get('tokenize', False)
                        )
                        conn.send({'result': result})
                    elif request.get('action') == 'tokenize':
                        tokens = tokenizer.tokenize(request['text'])
                        conn.send({'tokens': tokens})
                    elif request.get('action') == 'encode':
                        token_ids = tokenizer.encode(request['text'])
                        conn.send({'token_ids': token_ids})
                    elif request.get('action') == 'decode':
                        text = tokenizer.decode(request['token_ids'])
                        conn.send({'text': text})
                except EOFError:
                    break
                    
        except Exception as e:
            conn.send({'error': str(e)})
        finally:
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            conn.close()
        
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type (generator or verifier)."""
        pass
        
    @abstractmethod
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model initialization arguments."""
        pass
        
    def get_tokenizer(self):
        """Get the tokenizer from the model."""
        if self.process:
            # For process-based models, we need to get tokenizer info through the process
            # Since we can't directly access the tokenizer, we'll create a simple wrapper
            # that can handle basic tokenization needs
            return ProcessTokenizerWrapper(self.parent_conn)
        return self.model.get_tokenizer()
        
    def get_tokenizer_info(self):
        """Get tokenizer information for process-based models."""
        if self.process:
            self.parent_conn.send({'action': 'get_tokenizer_info'})
            result = self.parent_conn.recv()
            return result.get('tokenizer_info')
        return None
        
    def wake_up(self):
        """Wake up the model if it's in sleep mode."""
        if self.process:
            # Send wake up command to process
            self.parent_conn.send({'action': 'wake_up'})
            _ = self.parent_conn.recv()
        elif self.model and hasattr(self.model, 'wake_up'):
            self.model.wake_up()
            
    def sleep(self):
        """Put the model to sleep to save memory."""
        if self.process:
            # Send sleep command to process
            self.parent_conn.send({'action': 'sleep'})
            _ = self.parent_conn.recv()
        elif self.model and hasattr(self.model, 'sleep'):
            self.model.sleep()
            
    def shutdown(self):
        """Shutdown the model gracefully."""
        if self.process:
            try:
                # Send shutdown command to process
                self.parent_conn.send({'action': 'shutdown'})
                self.process.join(timeout=10)  # Add timeout
            except (BrokenPipeError, EOFError):
                # Process may have already terminated
                pass
            finally:
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
                self.parent_conn.close()
                self.process = None
        elif self.model:
            del self.model
        logger.info(f"{self.model_type.capitalize()} model shutdown complete")


class GeneratorVLLMModelWrapper(BaseVLLMModelWrapper):
    """VLLM model wrapper for generator models."""
    
    @property
    def model_type(self) -> str:
        return "generator"
        
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model initialization arguments for generator."""
        return {
            **self.config.generator_vllm_config,        
            "enable_sleep_mode": self.enable_sleep_mode,
            "spec_beam_extension": self.config.spec_beam_extension,
            "prefix_aware_scheduling": self.config.prefix_aware_scheduling,
        }

    def generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        if self.process:
            # Send generate request to process
            self.parent_conn.send({
                'action': 'generate',
                'prompts': prompts,
                'kwargs': kwargs
            })
            result = self.parent_conn.recv()
            if result.get('error'):
                raise RuntimeError(f"Failed to generate: {result['error']}")
            return result['result']
        else:
            results = self.model.generate_text(prompts, **kwargs)
            logger.info(f"Generator model generated {len(results)} results")
            return results


class VerifierVLLMModelWrapper(BaseVLLMModelWrapper):
    """VLLM model wrapper for verifier models."""
    
    @property
    def model_type(self) -> str:
        return "verifier"
        
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model initialization arguments for verifier."""
        override_pooler_config = {
            "pooling_type": "STEP", 
            "step_tag_id": 12902, 
            "returned_token_ids": [648, 387], 
            "softmax": True
        } if self.config.verifier_vllm_config["model"] == "peiyi9979/math-shepherd-mistral-7b-prm" else None
        return {
            **self.config.verifier_vllm_config,
            "enable_sleep_mode": self.enable_sleep_mode,
            "prefix_aware_scheduling": self.config.prefix_aware_scheduling,
            "task": "reward",
            "override_pooler_config": override_pooler_config,
        }

    def score(self, questions: List[str], outputs: List[List[str]], **kwargs) -> List[List[float]]:
        if self.process:
            # Send score request to process
            self.parent_conn.send({
                'action': 'score',
                'questions': questions,
                'outputs': outputs,
                'system_prompt': self.config.system_prompt,
                'kwargs': kwargs
            })
            result = self.parent_conn.recv()
            if result.get('error'):
                raise RuntimeError(f"Failed to score: {result['error']}")
            return result['result']
        else:
            scores = self.model.score_outputs(questions, outputs, self.config.system_prompt, **kwargs)
            return scores

class ProcessTokenizerWrapper:
    """Wrapper for tokenizer functionality when using process-based models."""
    
    def __init__(self, parent_conn):
        self.parent_conn = parent_conn
        
    def apply_chat_template(self, conversations, add_generation_prompt=True, continue_final_message=False, tokenize=False):
        """Apply chat template through the process."""
        self.parent_conn.send({
            'action': 'apply_chat_template',
            'conversations': conversations,
            'add_generation_prompt': add_generation_prompt,
            'continue_final_message': continue_final_message,
            'tokenize': tokenize
        })
        result = self.parent_conn.recv()
        return result.get('result')
        
    def tokenize(self, text):
        """Tokenize text through the process."""
        self.parent_conn.send({
            'action': 'tokenize',
            'text': text
        })
        result = self.parent_conn.recv()
        return result.get('tokens', [])
        
    def encode(self, text):
        """Encode text to token IDs through the process."""
        self.parent_conn.send({
            'action': 'encode',
            'text': text
        })
        result = self.parent_conn.recv()
        return result.get('token_ids', [])
    
    def decode(self, token_ids):
        """Decode token IDs to text through the process."""
        self.parent_conn.send({
            'action': 'decode',
            'token_ids': token_ids
        })
        result = self.parent_conn.recv()
        return result.get('text', '')


