from litellm.types.utils import ModelResponse


def prompt_user_to_continue(msg: str) -> bool:
    """Prompt the user to continue the process."""
    print(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------"
    )
    response = input(msg + " (y/q): ").strip().lower()
    if response == "q":
        raise SystemExit
    return response == "y"


def get_assistant_message_from_tool_call(
    model: str,
    completion: ModelResponse,
) -> None:
    if model == "fireworks_ai/accounts/fireworks/models/deepseek-v3":
        return {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }

    tool_call = completion.choices[0].message.tool_calls[0]
    return {
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
    # if model == "anthropic/claude-3-sonnet-20240229":
    #     # Documentation found here, under "Handling tool use and tool result content blocks"
    #     #
    #     # https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-of-tool-result-with-images
    #     return {
    #         "role": "assistant",
    #         "content": [
    #             {
    #                 "type": "tool_use",
    #                 "id": tool_call.id,
    #                 "name": tool_call.function.name,
    #                 "input": tool_call.function.arguments,
    #             }
    #         ],
    #     }
    # elif model == "gpt-4o-2024-08-06":
    #     return {
    #         "role": "tool",
    #         "tool_call_id": tool_call.id,
    #         # This should be the result of calling the function with the arguments provided?
    #         # Maybe stuff the test results in here?
    #         "content": tool_call.function.arguments,
    #         "name": tool_call.function.name,
    #     }
    # else:
    #     raise NotImplementedError(f"Model {model} not supported")


def get_user_message_from_tool_call(
    model: str,
    completion: ModelResponse,
    test_output: str,
    test_code: str,
    is_error: bool,
) -> None:
    if model == "fireworks_ai/accounts/fireworks/models/deepseek-v3":
        return {
            "role": "user",
            "content": f"""{test_code}

The translated code resulted in one or more errors while running tests, listed
below. The tests that ran are listed above. Modify the code produced above to ensure tests pass.

{test_output}""",
        }

    tool_call = completion.choices[0].message.tool_calls[0]
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": f"""{test_code}

The translated code resulted in one or more errors while running tests, listed
below. The tests that ran are listed above. Modify the code produced above to ensure tests pass.

{test_output}""",
        # "name": tool_call.function.name,
    }
    # if model == "anthropic/claude-3-sonnet-20240229":
    #     return {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "tool_result",
    #                 "tool_use_id": tool_call.id,
    #                 "content": test_output,
    #                 "is_error": is_error,
    #             }
    #         ],
    #     }
    # elif model == "gpt-4o-2024-08-06":
    #     return {
    #         "role": "tool",
    #         "tool_call_id": tool_call.id,
    #         "content": test_output,
    #         "name": tool_call.function.name,
    #     }
    # else:
    #     raise NotImplementedError(f"Model {model} not supported")
