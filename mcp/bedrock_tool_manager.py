from typing import Any, Dict, List, Callable


class BedrockConverseToolManager:
    def __init__(self):
        self._tools = {}
        self._name_mapping = {}

    def register_tool(self, name: str, func: Callable, description: str, input_schema: Dict):
        """
        도구를 등록합니다.

        Args:
            name: 도구 이름
            func: 도구 실행 함수
            description: 도구 설명
            input_schema: 입력 스키마
        """
        sanitized_name = name.replace('-', '_')
        self._name_mapping[sanitized_name] = name
        self._tools[sanitized_name] = {
            'function': func,
            'description': description,
            'input_schema': input_schema,
            'original_name': name
        }

    def get_tools(self) -> Dict[str, List[Dict]]:
        """
        Bedrock API에 전달할 도구 사양을 생성합니다.

        Returns:
            도구 사양 목록
        """
        tool_specs = []
        for sanitized_name, tool in self._tools.items():
            input_schema = tool['input_schema']
            if 'json' not in input_schema:
                input_schema = {'json': input_schema}

            # Set default values for required schema fields
            input_schema['json'].setdefault('type', 'object')
            input_schema['json'].setdefault('properties', {})
            input_schema['json'].setdefault('required', [])

            tool_specs.append({
                'toolSpec': {
                    'name': sanitized_name,
                    'description': tool['description'],
                    'inputSchema': input_schema
                }
            })

        return {'tools': tool_specs}

    async def execute_tool(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        도구 실행 요청을 처리합니다.

        Args:
            payload: 도구 실행 요청 페이로드

        Returns:
            도구 실행 결과
        """
        tool_use_id = payload['toolUseId']
        sanitized_name = payload['name']

        print(f"  • 사용 도구 : {sanitized_name}")
        if sanitized_name not in self._tools:
            return self._error_response(tool_use_id, f"Unknown tool: {sanitized_name}")

        try:
            tool_func = self._tools[sanitized_name]['function']
            original_name = self._tools[sanitized_name]['original_name']
            result = await tool_func(original_name, payload['input'])

            return {
                'toolUseId': tool_use_id,
                'content': [{'text': str(result)}],
                'status': 'success'
            }
        except Exception as e:
            return self._error_response(tool_use_id, str(e))

    def _error_response(self, tool_use_id: str, error_msg: str) -> Dict[str, Any]:
        """
        도구 실행 중 오류가 발생했을 때 표준화된 오류 응답을 생성합니다.

        Args:
            tool_use_id: 도구 사용 ID
            error_msg: 오류 메시지

        Returns:
            오류 응답
        """
        return {
            'toolUseId': tool_use_id,
            'content': [{'text': f"Error executing tool: {error_msg}"}],
            'status': 'error'
        }