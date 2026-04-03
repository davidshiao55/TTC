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

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fasttts",
    version="0.1.0",
    author="FastTTS Team",
    author_email="fasttts@example.com",
    description="Fast Test Time Search with async generator and verifier models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/fasttts",
    packages=find_packages(),
    py_modules=["fasttts", "core", "config"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "datasets",
        "accelerate",
        "safetensors",
        "tokenizers",
        "protobuf",
        "requests",
        "huggingface-hub",
        "fastapi",
        "uvicorn",
        "ray",
        "psutil",
        "rich",
        "typer",
        "python-dotenv",
        "python-json-logger",
        "prometheus-client",
        "prometheus-fastapi-instrumentator",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "watchfiles",
        "websockets",
        "uvloop",
        "httpx",
        "aiohttp",
        "msgspec",
        "msgpack",
        "xxhash",
        "fastrlock",
        "einops",
        "xformers",
        "triton",
        "numba",
        "llvmlite",
        "pycuda",
        "cupy-cuda12x",
        "compressed-tensors",
        "depyf",
        "cloudpickle",
        "dill",
        "diskcache",
        "multiprocess",
        "pebble",
        "word2number",
        "timeout-decorator",
        "python-dateutil",
        "py-cpuinfo",
        "pyarrow",
        "pybase64",
        "pycountry",
        "pytools",
        "pytz",
        "pyyaml",
        "pyzmq",
        "referencing",
        "regex",
        "rpds-py",
        "scipy",
        "sentencepiece",
        "shellingham",
        "siphash24",
        "sniffio",
        "starlette",
        "sympy",
        "tiktoken",
        "typing-inspection",
        "tzdata",
        "urllib3",
        "yarl",
        "zipp",
    ],
    extras_require={
        "dev": [
            "pytest==7.0.0",
            "pytest-asyncio==0.21.0",
            "black==22.0.0",
            "isort==5.0.0",
            "flake8==5.0.0",
            "mypy==1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fasttts=FastTTS.example:main",
        ],
    },
)
