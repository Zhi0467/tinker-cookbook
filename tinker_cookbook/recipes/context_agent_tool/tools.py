import logging
from abc import ABC, abstractmethod
from typing import Any
from tinker_cookbook.renderers import Message, ToolCall
# We import the interface, but our class will be much simpler

logger = logging.getLogger(__name__)

# This config isn't needed if the tool client itself does nothing
# @chz.chz 
# class ContextManageConfig:
#     type: str
class ToolClientInterface(ABC):
    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def invoke(self, tool_call: ToolCall) -> list[Message]: ...


class ContextToolClient(ToolClientInterface):
    """
    This class defines the schemas for context management tools.
    The actual execution logic for these tools lives in the
    environment's 'step()' method (e.g., in context_env.py),
    because they modify the environment's internal state (the history).
    """

    def __init__(self):
        # No setup needed
        pass

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Returns the list of tool definitions for the model.
        """
        return [
            {
                "name": "delete",
                "title": "Delete Context String",
                "description": "Deletes the first exact match of a specific string from the conversation history. Use this to remove redundant, incorrect, or unnecessary information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "string_to_delete": {
                            "type": "string",
                            "description": "The exact string to be deleted from the context. The match is case-sensitive and includes all whitespace.",
                        }
                    },
                    "required": ["string_to_delete"],
                },
                "outputSchema": {
                    "type": "string",
                    "description": "Returns a confirmation that the string was deleted.",
                },
            }
        ]

    async def invoke(self, tool_call: ToolCall) -> list[Message]:
        """
        This method will be called by the environment, but it won't
        do the actual work. It just formats the tool output message.
        The environment's 'step' method must handle the actual deletion/insertion.
        """
        tool_name = tool_call["name"]
        
        if tool_name == "delete":
            # Just validate the args.
            if "string_to_delete" in tool_call["args"]:
                content = f"Tool 'delete' called. The environment will now attempt to delete the string."
            else:
                content = "Error: 'string_to_delete' argument was missing."
            string_to_delete = tool_call["args"]["string_to_delete"]
            if (not isinstance(string_to_delete, str) 
                or not len(string_to_delete) > 0):
                return [
                    Message(
                        role="tool",
                        content="Error invoking delete tool: target must be a non-empty string",
                    )
                ]
            return [Message(role="tool", content=content)]
        
        else:
            raise ValueError(f"Invalid tool name: {tool_name}")