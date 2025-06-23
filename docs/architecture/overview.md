NCOS v11.5 Phoenix-Mesh Architecture

Overview

NCOS v11.5 "Phoenix-Mesh" is a next-generation LLM runtime that enables multiple specialized agents to operate within a single LLM session. This document outlines the core architectural components, data flows, and design principles of the system.

Core Concepts

Neural Agent Mesh

The Neural Agent Mesh is the central innovation of the Phoenix-Mesh architecture. It allows multiple virtual agents to operate within a single LLM session, sharing a unified token budget and memory context. This approach solves several key challenges:

Token Efficiency: By virtualizing agents within a single session, we avoid redundant context loading and token wastage.

Memory Coherence: All agents share access to the same memory space, ensuring consistent understanding.

Specialized Roles: Each agent can focus on a specific domain while operating in the broader shared context.

Dynamic Routing: The Unified Orchestrator can direct requests to the appropriate specialized agent.

Single-Session LLM Runtime

The single-session approach is a fundamental constraint that drives many architectural decisions:

All agent interactions occur within a single LLM session.

Token budget is managed globally across all agents.

Context is shared and must be optimized for relevance.

Memory references replace full context inclusion where possible.

Three-Tier Memory Architecture

The system implements a three-tier memory architecture:

L1: Session Memory (Redis)

Fastest access, limited persistence

Used for immediate context and short-term memory

Volatile, with optional persistence

L2: Vector Memory (FAISS)

Semantic search and retrieval

Used for knowledge bases and pattern recognition

Persisted across sessions

L3: Persistent Storage (PostgreSQL)

Long-term storage and complex relationships

Used for trade journals, user preferences, and system state

Fully persistent and queryable

System Components

Unified Orchestrator

The Unified Orchestrator (`core/orchestrators/unified_orchestrator.py`) is the central control component that:

Initializes the system and loads configurations

Manages the Neural Mesh Kernel

Coordinates agent lifecycle (loading, initialization, execution)

Handles session state and token budget management

Provides error recovery and health monitoring

Neural Mesh Kernel

The Neural Mesh Kernel (kernel.py) is responsible for:

Agent virtualization within the single LLM session

Message routing between agents

Action execution and result processing

Token allocation and monitoring

Memory access coordination

Session State

The Session State (session_state.py) tracks:

Current token usage and budget

Agent states and health

Execution history and queue

Memory namespaces and access

Agent Adapters

Agent Adapters provide a standardized interface for different agent implementations:

Base adapter for common functionality

Specialized adapters for different LLM providers

Adapters for non-LLM components (e.g., rule-based agents)

Memory Management

Memory management components include:

Vector Client for vector database interactions

Memory Manager for cross-tier memory operations

Namespace management for agent memory isolation

Data Flow

System Initialization

Load system configuration from config/phoenix.yaml

Initialize Unified Orchestrator

Set up Session State and Token Budget

Initialize Memory Systems (L1, L2, L3)

Load Agent Profiles from config/agent_profiles/

Initialize Neural Mesh Kernel

Register agents with the kernel

Agent Execution Flow

Orchestrator identifies the next action to execute

Token budget is allocated for the action

Neural Mesh Kernel routes the action to the appropriate agent

Agent executes the action and returns a result

Result is processed and stored in memory

Token usage is updated

Next action is determined

Memory Operations

Agent requests data from a specific namespace and tier

Memory Manager routes the request to the appropriate storage system

Data is retrieved and returned to the agent

Agent updates data if needed

Memory Manager handles persistence based on tier policies

Token Budget Management

Token budget management is critical for the single-session architecture:

Global token budget is defined in configuration

Unified Orchestrator allocates tokens to agents based on priority

Token usage is tracked for each agent and operation

Warning and critical thresholds trigger optimization strategies

Optimization includes context pruning, memory referencing, and agent prioritization

Extending the System

Adding New Agents

To add a new agent:

Create an agent profile YAML file in config/agent_profiles/

Implement the agent logic in src/agents/

Register the agent in config/phoenix.yaml

Customizing Memory Storage

The memory system can be customized by:

Implementing new storage backends

Defining custom namespaces and TTLs

Implementing specialized vector indexes

Adding New Capabilities

New capabilities can be added by:

Extending agent profiles with new capability definitions

Implementing the capability logic in the agent code

Updating triggers to invoke the new capabilities

Design Principles

The Phoenix-Mesh architecture is guided by several key principles:

Single-Session Constraint: All design decisions respect the token and context limitations of a single LLM session.

Modular Components: System components are modular and replaceable.

Configuration-Driven: Behavior is defined in configuration rather than code where possible.

Resilience: The system can recover from errors and maintain state.

Observability: All operations are logged and monitored.

Extensibility: The system can be extended with new agents and capabilities.

Conclusion

The NCOS v11.5 Phoenix-Mesh architecture represents a significant advancement in LLM runtime design, enabling complex multi-agent workflows within the constraints of a single LLM session. By virtualizing agents through the Neural Agent Mesh, the system achieves both specialization and coherence while optimizing token usage and memory management.