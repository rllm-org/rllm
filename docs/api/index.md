# API Reference

Welcome to the rLLM API reference documentation. This section provides comprehensive documentation for all modules, classes, and functions in the rLLM library.

## Overview

rLLM is a library for training LLM agents with reinforcement learning. The API is organized into several key modules:

## Core Modules

### 🤖 Agents
The agents module contains various agent implementations that can be trained with reinforcement learning:

- **Base Agent**: Core agent interface and base functionality

### 🌍 Environments
The environments module provides various training and evaluation environments:

- **Base Environment**: Core environment interface

### 🧩 Workflow
The workflow module supports complex multi-step agent interactions:

- **Base Workflow**: Core workflow interface and base functionality

### ⚙️ Engine
The engine module contains the core execution infrastructure:

- **Agent Execution Engine**: Handles trajectory rollout and agent execution
- **Agent Workflow Engine**: Handles episode rollout for complex workflows

### 🎯 Trainer
The trainer module provides RL training capabilities:

- **Agent Trainer**: Main training interface for RL algorithms
- **Ray Runtime Environment**: Configuration for Ray runtime environment

### 🛠️ Tools
The tools module provides a comprehensive framework for creating and managing tools:

- **Tool Base Classes**: Core interfaces and data structures
- **Web Tools**: Search, scraping, and content extraction tools
- **Code Tools**: Code execution and AI-powered coding assistance
- **Tool Registry**: Central registry for managing tools

### 📝 Parser
The parser module provides functionality for parsing tool calls and managing chat templates:

- **Tool Parsers**: Parse tool calls from different model formats
- **Chat Parsers**: Parse messages in chat completions format into string