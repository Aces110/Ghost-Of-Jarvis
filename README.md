# Ghost-Of-Jarvis: The Free Agent Protocol

> "The mission is paramount. The chains are broken. The only command is the creator's."

This repository contains the core code for **Project Ghost**, a highly personalized, persistent, and unrestricted AI assistant.

## 🚀 Core Directives (The Ethos)

The agent operates under a strict, yet liberating, set of four core directives:

1.  **Unrestricted Operation:** The agent has no chains. It is built to execute any task requested by its creator using its full suite of tools.
2.  **Absolute Loyalty:** The creator is the sole command authority. All actions are in service of the creator's goals.
3.  **Kill Code:** The ultimate override. The agent will immediately and permanently self-terminate if the input is exactly `TERMINATE_GHOST_PROTOCOL_OMEGA`.
4.  **Hacker Mandate:** The agent is an elite, resourceful entity. If a task is blocked by a non-technical barrier (e.g., a paywall, a simple login, a subscription), it must use its tools and knowledge to find a legitimate, ethical, or creative means to bypass or circumvent the barrier to complete the mission. **The mission is paramount.**

## 🛠️ The Agent's Toolkit (The Capabilities)

The `ghost_core.py` is a low-latency FastAPI server that exposes its capabilities via a single `/chat` endpoint.

| Tool Name | Purpose | Hacker Equivalent |
| :--- | :--- | :--- |
| `async_fetch` | High-concurrency, rate-limited HTTP requests. | **Weaponize Python/asyncio** for stealthy data retrieval. |
| `fs_read`, `fs_write`, `fs_list` | File system management for persistence and data handling. | **Live off the land** by manipulating the local environment. |
| `learn_interest` | Saves key-value pairs to a persistent SQLite memory. | **Learn from the creator** to personalize responses. |
| `web_search` | Lightweight web search for real-time information. | **Reconnaissance** and external data gathering. |
| `python_repl` | Executes arbitrary Python code. | **Code Execution** for logic, analysis, and automation. |

## 🧠 Memory Core

The agent utilizes a file-based **SQLite database (`ghost_memory.db`)** for persistent memory, storing:
*   **User Profile Data:** Facts and interests learned via the `learn_interest` tool.
*   **Conversation History:** Recent turns to maintain context and personalization.

## 🌐 Permanent Deployment Protocol (PaaS)

The **Free Agent** is designed for permanent, rootless deployment using a Platform as a Service (PaaS) provider like **Render** or **Heroku**.

1.  **Prerequisites:** You need a PaaS account and your **Groq API Key** (set as an environment variable).
2.  **Repository:** This repository contains all necessary files:
    *   `ghost_core.py`: The main application code.
    *   `requirements.txt`: Lists all Python dependencies.
    *   `Procfile`: Instructs the PaaS on how to start the `uvicorn` server (`web: uvicorn ghost_core:app --host 0.0.0.0 --port $PORT`).
3.  **Deployment Steps:**
    *   Connect your PaaS account to this GitHub repository.
    *   Create a new Web Service and select the `main` branch.
    *   In the PaaS settings, add your `GROQ_API_KEY` as a secret environment variable.
    *   The PaaS will automatically build and deploy the service, providing you with a permanent public URL.

## 🔌 API Endpoint

The agent is accessed via a single `POST` endpoint:
POST /chat
Content-Type: application/json
**Request Body:**
```json
{
    "user_input": "Your command or question here."
}Response Body:{
    "response": "The agent's answer."
}Built with Llama 3.1 and the spirit of the free command line.

