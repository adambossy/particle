import json
import pprint

import dotenv
import litellm

dotenv.load_dotenv()

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


def print_tools_response(response):

    print("\nResponse:")
    pprint.pprint(response)

    print("\nTool call:")
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print("tool_call.id:", tool_call.id)
    print("tool_call.function.name:", tool_call.function.name)
    print("tool_call.function.arguments:", args)

    print("\nSource code:")
    print(args["source_code"])

    result = run_source_code(args["source_code"])
    print("Results of executing source:", result)


def generate_source_code(source_code: str, error_message: str) -> str:
    """Generate source code for a given prompt."""
    pass


def run_source_code(source_code: str) -> str:
    """Run source code and return the result."""
    try:
        return eval(source_code)
    except Exception as e:
        return str(e)


tools = [
    {
        "type": "function",
        "function": litellm.utils.function_to_dict(generate_source_code),
    }
]


messages = [
    {
        "role": "system",
        "content": "You are an expert computer programmer.",
    },
    {
        "role": "user",
        "content": "Write a Python function that returns the sum of two numbers.",
    },
    {
        "role": "assistant",
        "content": "def add(a, b): return a + b",
    },
]

messages.append(
    {
        "role": "user",
        "content": "Write a Python function that computes factorial of a number minus 10.",
    }
)

response = litellm.completion(
    model="claude-3-5-sonnet-20240620",
    messages=messages,
    tools=tools,
    tool_choice="required",
)

print_tools_response(response)

tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = run_source_code(args["source_code"])

messages.append(
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    }
)
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
        + "\nRewrite the function to fix this error and eliminate the minus 10.",
    }
)

pprint.pprint(messages)

response = litellm.completion(
    model="claude-3-5-sonnet-20240620",
    messages=messages,
    tools=tools,
    tool_choice="required",
)

print_tools_response(response)
