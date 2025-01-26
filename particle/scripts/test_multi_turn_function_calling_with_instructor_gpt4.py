import json
import pprint
import uuid

import dotenv
import instructor
import litellm
from pydantic import BaseModel

dotenv.load_dotenv()

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

client = instructor.from_litellm(litellm.completion)


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


class GenerateSourceCodeResponse(BaseModel):
    """Generate source code for a given prompt."""

    source_code: str
    error_message: str


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

trace_id = "test-multi-turn-function-calling-gpt4-" + str(uuid.uuid4())


response, completion = client.chat.completions.create_with_completion(
    model="gpt-4o-2024-08-06",
    response_model=GenerateSourceCodeResponse,
    messages=messages,
    metadata={
        "trace_id": trace_id,
    },
)

print(response)
print_tools_response(completion)

tool_call = completion.choices[0].message.tool_calls[0]
source_code = response.source_code
result = run_source_code(source_code)


# Instructor fails to convert the tool_calls message correctly 
# 
#   File "/Users/adambossy/code/particle/particle/scripts/test_multi_turn_function_calling_with_retries_gpt4.py", line 131, in <module>
#     response, completion = client.chat.completions.create_with_completion(
#   File "/Users/adambossy/Library/Caches/pypoetry/virtualenvs/language-translation-2S16z6H9-py3.10/lib/python3.10/site-packages/instructor/client.py", line 321, in create_with_completion
#     model = self.create_fn(
#   File "/Users/adambossy/Library/Caches/pypoetry/virtualenvs/language-translation-2S16z6H9-py3.10/lib/python3.10/site-packages/instructor/patch.py", line 187, in new_create_sync
#     response_model, new_kwargs = handle_response_model(
#   File "/Users/adambossy/Library/Caches/pypoetry/virtualenvs/language-translation-2S16z6H9-py3.10/lib/python3.10/site-packages/instructor/process_response.py", line 755, in handle_response_model
#     new_kwargs["messages"] = convert_messages(
#   File "/Users/adambossy/Library/Caches/pypoetry/virtualenvs/language-translation-2S16z6H9-py3.10/lib/python3.10/site-packages/instructor/multimodal.py", line 336, in convert_messages
#     content = message["content"] or []
# KeyError: 'content'
# 
# (Pdb) message
# {'role': 'assistant', 'tool_calls': [{'id': 'call_BzRKnteFNXdYLCcYxGEAoX2m', 'type': 'function', 'function': {'name': 'GenerateSourceCodeResponse', 'arguments': '{"source_code":"def factorial_minus_ten(n):\\n    \\"\\"\\"\\n    Compute the factorial of a number \'n\' and then subtract 10 from it.\\n\\n    Parameters:\\n    n (int): A non-negative integer for which to compute the factorial.\\n\\n    Returns:\\n    int: The result of factorial(n) - 10.\\n    \\"\\"\\"\\n    if n < 0:\\n        raise ValueError(\\"Input must be a non-negative integer\\")\\n    \\n    factorial = 1\\n    for i in range(2, n + 1):\\n        factorial *= i\\n    \\n    return factorial - 10","error_message":""}'}}]}
# (

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
        "content": result + "\nRewrite the function to fix this error.",
    }
)

pprint.pprint(messages)

response, completion = client.chat.completions.create_with_completion(
    model="gpt-4o-2024-08-06",
    response_model=GenerateSourceCodeResponse,
    messages=messages,
    metadata={
        "trace_id": trace_id,
    },
)

print(response)
print_tools_response(completion)
result = run_source_code(source_code)
