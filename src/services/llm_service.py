"""
LLM 服务封装
处理与 OpenAI API 的交互
"""
from openai import AsyncOpenAI
from typing import List, Dict, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 调用服务"""
    
    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-72B-Instruct", base_url: str = None):
        """
        初始化 LLM 服务，支持硅基流动平台
        Args:
            api_key: API Key
            model: 使用的模型名称
            base_url: API Base URL (硅基流动平台)
        """
        if base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"✅ LLM 服务初始化成功，模型: {model}, base_url: {base_url}")
        else:
            self.client = AsyncOpenAI(api_key=api_key)
            logger.info(f"✅ LLM 服务初始化成功，模型: {model}")
        self.model = model
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """
        流式生成对话，兼容 reasoning_content 字段
        """
        try:
            logger.info(f"🤖 开始流式生成，消息数: {len(messages)}")
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        yield delta.reasoning_content
            logger.info("✅ 流式生成完成")
                    
        except Exception as e:
            logger.error(f"❌ LLM 生成失败: {e}")
            yield f"\n\n[错误: {str(e)}]"
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        非流式生成对话
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
        
        Returns:
            生成的回复文本
        """
        try:
            logger.info(f"🤖 开始生成回复，消息数: {len(messages)}")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            logger.info(f"✅ 生成完成，长度: {len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"❌ LLM 生成失败: {e}")
            return f"抱歉，生成回答时出错: {str(e)}"
