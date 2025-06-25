import asyncio
from contextlib import AsyncExitStack
from typing import Any, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from bedrock_converse import BedrockConverse
from bedrock_tool_manager import BedrockConverseToolManager


class MCPClient:
    def __init__(self, model_id: str):
        """
        MCP 클라이언트를 초기화합니다.

        Args:
            model_id: 사용할 모델 ID
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model_id = model_id
        self.converse = BedrockConverse(model_id)
        self.converse.tools = BedrockConverseToolManager()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    async def connect_stdio(self, server_params: StdioServerParameters):
        """
        MCP 서버에 연결합니다.

        Args:
            server_params: 서버 파라미터
        """
        self._client = stdio_client(server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()

    async def get_available_tools(self) -> List[Any]:
        """
        사용 가능한 도구 목록을 조회합니다.

        Returns:
            도구 목록
        """
        if not self.session:
            raise RuntimeError("Not connected to server")
        response = await self.session.list_tools()
        return response.tools

    async def call_tool(self, tool_name: str, params: dict) -> Any:
        """
        도구를 호출합니다.

        Args:
            tool_name: 도구 이름
            params: 도구 파라미터

        Returns:
            도구 호출 결과
        """
        if not self.session:
            raise RuntimeError("Not connected to server")
        return await self.session.call_tool(tool_name, params)


async def main():
    """
    메인 함수
    """
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server/tools.py"],
        env=None
    )

    async with MCPClient(model_id) as mcp_client:
        try:
            await mcp_client.connect_stdio(server_params)
            tools = await mcp_client.get_available_tools()

            for tool in tools:
                mcp_client.converse.tools.register_tool(
                    name=tool.name,
                    func=mcp_client.call_tool,
                    description=tool.description,
                    input_schema={'json': tool.inputSchema}
                )

            print("Available tools:")
            for tool in tools:
                print(f"  • {tool.name}: {tool.description}")

            while True:
                try:
                    user_prompt = input("\nUser: ")
                    if not user_prompt.strip():
                        continue

                    print("Thinking...")
                    response = await mcp_client.converse.invoke_with_prompt(user_prompt)
                    print(f"Assistant: {response}")

                except KeyboardInterrupt:
                    break

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())