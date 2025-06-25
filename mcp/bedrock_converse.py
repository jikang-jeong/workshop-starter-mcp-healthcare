from typing import List, Dict, Any
import asyncio
import boto3
import time


class BedrockConverse:
    def __init__(self, model_id: str, region: str = 'us-west-2'):
        """
        Bedrock API와 통신하는 클래스를 초기화합니다.

        Args:
            model_id: 사용할 모델 ID
            region: AWS 리전
        """
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.messages: List[Dict[str, Any]] = []
        self.tools = None

    async def invoke_with_prompt(self, prompt: str) -> str:
        """
        텍스트 프롬프트로 모델을 호출합니다.

        Args:
            prompt: 사용자 프롬프트

        Returns:
            모델 응답
        """
        content = [{'text': prompt}]
        return await self.invoke(content)

    async def invoke(self, content: List[Dict[str, str]]) -> str:
        """
        모델을 호출합니다.

        Args:
            content: 메시지 내용

        Returns:
            모델 응답
        """
        self.messages.append({
            "role": "user",
            "content": content
        })
        await asyncio.sleep(0.5)  # throttling exception 방지를 위해 0.5초 대기
        response = self._get_converse_response()
        return await self._handle_response(response)

    def _get_converse_response(self) -> Dict:
        """
        Bedrock API를 호출하여 응답을 받습니다.

        Returns:
            API 응답
        """
        payload = {
            "modelId": self.model_id,
            "messages": self.messages,
            "system": [{
                           "text": "You are a helpful assistant. If there is a return value that includes a URL (such as an image), the response must include both the message and the URL together. "}],
            "inferenceConfig": {
                "maxTokens": 8192,
                "temperature": 0.7,
            },
            "toolConfig": self.tools.get_tools()
        }

        # 재시도 로직 추가
        max_retries = 5
        retry_delay = 1  # 초 단위

        for attempt in range(max_retries):
            try:
                return self.client.converse(**payload)
            except self.client.exceptions.ThrottlingException as e:
                if attempt < max_retries - 1:
                    print(
                        f"Throttling detected, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 지수 백오프
                else:
                    raise e
            except Exception as e:
                raise e

    async def _handle_response(self, response: Dict) -> str:
        """
        모델 응답을 처리합니다.

        Args:
            response: 모델 응답

        Returns:
            처리된 응답 텍스트
        """
        self.messages.append(response['output']['message'])
        stop_reason = response['stopReason']

        if stop_reason in ['end_turn', 'stop_sequence']:
            try:
                return response['output']['message']['content'][0]['text']
            except (KeyError, IndexError):
                return ''

        elif stop_reason == 'tool_use':
            tool_response = []
            for content_item in response['output']['message']['content']:
                if 'toolUse' in content_item:
                    tool_request = {
                        "toolUseId": content_item['toolUse']['toolUseId'],
                        "name": content_item['toolUse']['name'],
                        "input": content_item['toolUse']['input']
                    }
                    print(f"  • 도구 호출시 질의문 : {content_item['toolUse']['input']}")
                    tool_result = await self.tools.execute_tool(tool_request)
                    tool_response.append({'toolResult': tool_result})

            return await self.invoke(tool_response)

        return ''