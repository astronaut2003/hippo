import os
import logging
import traceback
from typing import List, Dict, Any, Optional
from openai import OpenAI
import re

# 添加tiktoken支持
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken未安装，将使用简单估算方法。建议安装: pip install tiktoken")


class LLMClient:
    """统一的大模型客户端接口，支持DeepSeek、GPT、GPT-CloseAI等多个提供商"""
    
    # API提供商配置
    API_PROVIDERS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "supports_json_format": True
        },
        "cstcloud":{
            "base_url": "https://uni-api.cstcloud.cn/v1",
            "api_key_env": "CSTCLOUD_API_KEY",
            "supports_json_format": True
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "supports_json_format": True
        },
        "openai-proxy": {  # closeai中转
            "base_url": "https://api.openai-proxy.org/v1",
            "api_key_env": "CLOSEAI_API_KEY",  # 独立的环境变量
            "fallback_env": "OPENAI_API_KEY",   # 备用环境变量
            "supports_json_format": True
        }
    }
    
    # 模型配置（包含提供商信息）
    MODEL_CONFIGS = {
        # DeepSeek 系列
        "deepseek-chat": {
            "provider": "deepseek",
            "context_length": 131072,    # 128K
            "max_output": 8192,
            "default_output": 4096,
            "encoding": "cl100k_base"
        },
        "deepseek-reasoner": {
            "provider": "deepseek",
            "context_length": 131072,    # 128K
            "max_output": 65536,
            "default_output": 32768,
            "encoding": "cl100k_base"
        },
        
        # 中国科技云大模型API
        "deepseek-r1:671b-0528": {
            "provider": "cstcloud",
            "context_length": 65536,
            "max_output": 8192,
            # "default_output": 4096,
            # 修改默认输出为8192，更适合数据处理场景
            "default_output": 8192,
            "encoding": "cl100k_base"
        },
        "deepseek-v3:671b": {
            "provider": "cstcloud",
            "context_length": 65536,
            "max_output": 8192,
            "default_output": 4096,
            "encoding": "cl100k_base"
        },
        
        # OpenAI GPT 系列
        "gpt-4o-mini": {
            "provider": "openai",
            "context_length": 131072,    # 128K
            "max_output": 16384,
            "default_output": 8192,
            "encoding": "o200k_base"
        },
        "gpt-4o": {
            "provider": "openai",
            "context_length": 131072,    # 128K
            "max_output": 16384,
            "default_output": 8192,
            "encoding": "o200k_base"
        },
        "gpt-4-turbo": {
            "provider": "openai",
            "context_length": 131072,
            "max_output": 4096,
            "default_output": 2048,
            "encoding": "o200k_base"
        },
        
        # CloseAI中转GPT（使用相同配置，不同提供商）
        "gpt-4o-mini-closeai": {
            "provider": "openai-proxy",
            "context_length": 131072,
            "max_output": 16384,
            "default_output": 8192,
            "encoding": "o200k_base",
            "actual_model": "gpt-4o-mini"  # 实际调用的模型名
        },
        "gpt-4o-closeai": {
            "provider": "openai-proxy",
            "context_length": 131072,
            "max_output": 16384,
            "default_output": 8192,
            "encoding": "o200k_base",
            "actual_model": "gpt-4o"
        }
    }
    
    def __init__(self, 
             model_name: str = "deepseek-reasoner",
             api_key: Optional[str] = None,
             base_url: Optional[str] = None,
             max_context_ratio: float = 0.85):
        """
        初始化LLM客户端
        
        Args:
            model_name: 模型名称
            api_key: API密钥（可选，默认从环境变量读取）
            base_url: API基础URL（可选，默认根据模型自动选择）
            max_context_ratio: 上下文占总长度的比例
        """
        # 1. 首先初始化 logger（因为其他方法需要用到）
        self.logger = logging.getLogger(__name__)
        
        # 2. 设置基本参数
        self.model_name = model_name
        self.max_context_ratio = max_context_ratio
        
        # 3. 获取模型配置
        self.model_config = self.MODEL_CONFIGS.get(model_name)
        if not self.model_config:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {list(self.MODEL_CONFIGS.keys())}")
        
        # 4. 确定API提供商
        self.provider = self.model_config["provider"]
        self.provider_config = self.API_PROVIDERS.get(self.provider)
        if not self.provider_config:
            raise ValueError(f"不支持的API提供商: {self.provider}")
        
        # 5. 设置API密钥（此时 logger 已经初始化，可以安全调用 _get_api_key）
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            env_vars = [self.provider_config["api_key_env"]]
            if "fallback_env" in self.provider_config:
                env_vars.append(self.provider_config["fallback_env"])
            raise ValueError(
                f"未找到API密钥。请设置以下任一环境变量: {', '.join(env_vars)} "
                f"或传入api_key参数"
            )
        
        # 6. 设置base_url（优先使用传入的，否则使用默认）
        self.base_url = base_url or self.provider_config["base_url"]
        
        # 7. 获取实际调用的模型名（处理closeai等中转情况）
        self.actual_model = self.model_config.get("actual_model", model_name)
        
        # 8. 设置上下文参数
        self.context_length = self.model_config["context_length"]
        self.max_output_tokens = self.model_config["max_output"]
        self.default_output_tokens = self.model_config["default_output"]
        self.max_context_tokens = int(self.context_length * max_context_ratio)
        
        # 9. 是否支持JSON格式
        self.supports_json_format = self.provider_config["supports_json_format"]
        
        # 10. 初始化tokenizer
        self.tokenizer = self._init_tokenizer()
        
        # 11. 初始化客户端
        self.client = self._init_client()
        
        # 12. 记录初始化信息
        self._log_initialization()
    
    def _get_api_key(self) -> Optional[str]:
        """
        从环境变量获取API密钥
        支持主环境变量和备用环境变量（用于兼容性）
        
        Returns:
            API密钥，如果未找到则返回None
        """
        # 获取主环境变量名
        primary_env = self.provider_config["api_key_env"]
        
        # 尝试从主环境变量获取
        api_key = os.getenv(primary_env)
        if api_key:
            self.logger.debug(f"✅ 从环境变量 {primary_env} 读取API密钥")
            return api_key
        
        # 如果有备用环境变量，尝试从备用环境变量获取
        if "fallback_env" in self.provider_config:
            fallback_env = self.provider_config["fallback_env"]
            api_key = os.getenv(fallback_env)
            if api_key:
                self.logger.info(
                    f"⚠️  主环境变量 {primary_env} 未找到，"
                    f"使用备用环境变量 {fallback_env}"
                )
                return api_key
        
        # 都未找到
        self.logger.warning(f"❌ 未找到环境变量 {primary_env}")
        if "fallback_env" in self.provider_config:
            self.logger.warning(f"   也未找到备用环境变量 {self.provider_config['fallback_env']}")
        
        return None
    
    def _init_tokenizer(self):
        """初始化tokenizer"""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        try:
            encoding_name = self.model_config["encoding"]
            tokenizer = tiktoken.get_encoding(encoding_name)
            self.logger.debug(f"使用tiktoken编码器: {encoding_name}")
            return tokenizer
        except Exception as e:
            self.logger.warning(f"tiktoken初始化失败: {e}，使用简单估算")
            return None
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.logger.debug(f"OpenAI客户端初始化成功: {self.base_url}")
            return client
        except Exception as e:
            self.logger.error(f"OpenAI客户端初始化失败: {e}")
            raise
    
    def _log_initialization(self):
        """记录初始化信息"""
        self.logger.info(f"🤖 初始化LLM客户端")
        self.logger.info(f"   模型: {self.model_name}")
        self.logger.info(f"   提供商: {self.provider}")
        self.logger.info(f"   API地址: {self.base_url}")
        self.logger.info(f"   实际模型: {self.actual_model}")
        self.logger.info(f"📏 上下文长度: {self.context_length:,}, 可用: {self.max_context_tokens:,}")
        self.logger.info(f"📤 最大输出: {self.max_output_tokens:,}, 默认: {self.default_output_tokens:,}")
        self.logger.info(f"🔧 Token计算: {'tiktoken' if self.tokenizer else '简单估算'}")
        self.logger.info(f"📝 JSON格式支持: {'是' if self.supports_json_format else '否'}")
    
    def count_tokens(self, text: str) -> int:
        """
        计算token数量
        优先使用tiktoken，fallback到简化估算
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # 简化的token估算
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        other_chars = len(text) - chinese_chars - english_chars
        
        estimated_tokens = int(
            chinese_chars * 0.6 +
            english_chars * 0.3 +
            other_chars * 0.4
        )
        
        return max(estimated_tokens, 1)
    
    def truncate_context(self, context_text: str, max_tokens: Optional[int] = None) -> str:
        """
        简化的上下文截断
        保留最后的内容（最新的对话通常最重要）
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        current_tokens = self.count_tokens(context_text)
        
        if current_tokens <= max_tokens:
            return context_text
        
        # 按行截断，从后往前保留
        lines = context_text.split('\n')
        selected_lines = []
        current_tokens = 0
        
        for line in reversed(lines):
            line_tokens = self.count_tokens(line + '\n')
            if current_tokens + line_tokens <= max_tokens:
                selected_lines.insert(0, line)
                current_tokens += line_tokens
            else:
                break
        
        result = '\n'.join(selected_lines)
        final_tokens = self.count_tokens(result)
        
        if final_tokens != current_tokens:
            self.logger.info(f"📏 上下文截断: {self.count_tokens(context_text)} -> {final_tokens} tokens")
        
        return result
    
    def generate_answer(self, 
                       prompt: str, 
                       max_tokens: Optional[int] = None,
                       temperature: float = 0.1,
                       generate_strategy: str = "default",
                       json_format: bool = False,
                       **kwargs) -> str:
        """
        生成答案
        
        Args:
            prompt: 完整提示词
            max_tokens: 最大输出token数
            temperature: 温度参数
            generate_strategy: 生成策略 ("default", "max")
            json_format: 是否要求JSON格式输出
            **kwargs: 其他API参数
            
        Returns:
            生成的答案
        """
        
        # 设置输出token数
        if max_tokens is None:
            if generate_strategy == "max":
                max_tokens = self.max_output_tokens
            else:
                max_tokens = self.default_output_tokens
        else:
            max_tokens = min(max_tokens, self.max_output_tokens)
        
        # 确保prompt不超过上下文限制
        prompt_tokens = self.count_tokens(prompt)
        max_prompt_tokens = self.context_length - max_tokens - 100
        
        if prompt_tokens > max_prompt_tokens:
            self.logger.warning(f"⚠️ Prompt过长 ({prompt_tokens} > {max_prompt_tokens})，截断中...")
            prompt = self.truncate_context(prompt, max_prompt_tokens)
            prompt_tokens = self.count_tokens(prompt)
            self.logger.info(f"📏 Prompt截断后: {prompt_tokens} tokens")
        
        try:
            # 构建请求参数
            request_params = {
                "model": self.actual_model,  # 使用实际模型名
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # 添加其他支持的参数
            supported_params = ['frequency_penalty', 'presence_penalty', 'top_p', 'stop']
            for param in supported_params:
                if param in kwargs:
                    request_params[param] = kwargs[param]
            
            # JSON格式输出支持（统一处理）
            if json_format and self.supports_json_format:
                request_params["response_format"] = {"type": "json_object"}
                # 确保prompt中包含JSON指令
                if "json" not in prompt.lower():
                    prompt += "\n\nPlease respond in JSON format."
                    request_params["messages"] = [{"role": "user", "content": prompt}]
                self.logger.debug("✅ 已启用JSON格式输出")
            elif json_format and not self.supports_json_format:
                self.logger.warning(f"⚠️ 模型 {self.model_name} 不支持JSON格式，将忽略json_format参数")
            
            self.logger.debug(f"🚀 发送请求: {prompt_tokens} tokens -> max {max_tokens} tokens")
            self.logger.debug(f"📡 API: {self.base_url}, 模型: {self.actual_model}")
            
            response = self.client.chat.completions.create(**request_params)
            
            answer = response.choices[0].message.content.strip()
            
            # 记录token使用
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                self.logger.debug(f"📊 Token使用: 输入={usage.prompt_tokens}, "
                                f"输出={usage.completion_tokens}, "
                                f"总计={usage.total_tokens}")
            else:
                estimated_output = self.count_tokens(answer)
                self.logger.debug(f"📊 Token估算: 输入≈{prompt_tokens}, 输出≈{estimated_output}")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"❌ 生成失败: {e}")
            self.logger.debug(f"Provider: {self.provider}, Model: {self.model_name}, Base URL: {self.base_url}")
            return f"生成失败: {str(e)}"
    
    def batch_generate(self, 
                      prompts: List[str], 
                      max_tokens: Optional[int] = None,
                      temperature: float = 0.1,
                      **kwargs) -> List[str]:
        """批量生成答案"""
        results = []
        
        self.logger.info(f"🔄 批量生成开始: {len(prompts)} 个请求")
        
        for i, prompt in enumerate(prompts, 1):
            self.logger.debug(f"处理 {i}/{len(prompts)}")
            answer = self.generate_answer(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            results.append(answer)
        
        self.logger.info(f"✅ 批量生成完成")
        return results
    
    def get_context_info(self) -> Dict[str, Any]:
        """获取LLM上下文配置信息"""
        return {
            "model_name": self.model_name,
            "actual_model": self.actual_model,
            "provider": self.provider,
            "base_url": self.base_url,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "default_output_tokens": self.default_output_tokens,
            "max_context_tokens": self.max_context_tokens,
            "tokenizer_available": self.tokenizer is not None,
            "encoding": self.model_config.get("encoding", "unknown"),
            "supports_json_format": self.supports_json_format
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型配置信息（别名方法）"""
        return self.get_context_info()
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """分析文本的token信息"""
        total_tokens = self.count_tokens(text)
        
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        return {
            "total_tokens": total_tokens,
            "character_count": len(text),
            "chinese_chars": chinese_chars,
            "english_chars": english_chars,
            "tokens_per_char": total_tokens / len(text) if text else 0,
            "fits_in_context": total_tokens <= self.max_context_tokens,
            "usage_ratio": total_tokens / self.context_length,
            "can_process": total_tokens <= (self.context_length - self.default_output_tokens)
        }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, List[str]]:
        """列出所有可用的模型，按提供商分组"""
        models_by_provider = {}
        
        for model_name, config in cls.MODEL_CONFIGS.items():
            provider = config["provider"]
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append(model_name)
        
        return models_by_provider
    
    @classmethod
    def get_provider_info(cls, provider: str) -> Optional[Dict[str, Any]]:
        """获取API提供商信息"""
        return cls.API_PROVIDERS.get(provider)
    
    @classmethod
    def list_required_env_vars(cls) -> Dict[str, List[str]]:
        """列出所有提供商需要的环境变量"""
        env_vars = {}
        
        for provider, config in cls.API_PROVIDERS.items():
            vars_list = [config["api_key_env"]]
            if "fallback_env" in config:
                vars_list.append(f"{config['fallback_env']} (备用)")
            env_vars[provider] = vars_list
        
        return env_vars


# 便捷函数
def create_llm_client(model: str = "deepseek-reasoner", 
                     api_key: Optional[str] = None,
                     base_url: Optional[str] = None) -> LLMClient:
    """创建LLM客户端的便捷函数"""
    return LLMClient(model_name=model, api_key=api_key, base_url=base_url)


def create_deepseek_client(model: str = "deepseek-chat", 
                          api_key: Optional[str] = None) -> LLMClient:
    """创建DeepSeek客户端的便捷函数（向后兼容）"""
    return LLMClient(model_name=model, api_key=api_key)


def estimate_tokens(text: str) -> int:
    """快速估算token数的独立函数"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    other_chars = len(text) - chinese_chars - english_chars
    
    return int(chinese_chars * 0.6 + english_chars * 0.3 + other_chars * 0.4)


# 使用示例
if __name__ == "__main__":
    import time
    import sys
    
    print("=" * 80)
    print("LLM客户端测试工具")
    print("=" * 80)
    
    # 列出所有可用模型
    print("\n📋 可用模型:")
    models = LLMClient.list_available_models()
    all_models = []
    model_index = 1
    for provider, model_list in models.items():
        print(f"\n{provider}:")
        for model in model_list:
            print(f"  [{model_index}] {model}")
            all_models.append(model)
            model_index += 1
    
    # 选择模型
    print("\n" + "=" * 80)
    model_choice = input(f"请选择要测试的模型 (1-{len(all_models)}, 直接回车默认使用 deepseek-v3:671b): ").strip()
    
    if model_choice == "":
        selected_model = "deepseek-v3:671b"
        print(f"使用默认模型: {selected_model}")
    elif model_choice.isdigit() and 1 <= int(model_choice) <= len(all_models):
        selected_model = all_models[int(model_choice) - 1]
        print(f"已选择模型: {selected_model}")
    else:
        print(f"❌ 无效选择，使用默认模型: deepseek-v3:671b")
        selected_model = "deepseek-v3:671b"
    
    # 输入测试问题
    print("\n" + "=" * 80)
    test_prompt = input("请输入测试问题 (直接回车使用默认问题): ").strip()
    if test_prompt == "":
        test_prompt = "请简单介绍一下人工智能。"
        print(f"使用默认问题: {test_prompt}")
    
    # 开始测试
    print("\n" + "=" * 80)
    print(f"测试模型: {selected_model}")
    print("=" * 80)
    
    try:
        # 初始化客户端
        print("\n⏳ 正在初始化客户端...")
        init_start = time.time()
        client = LLMClient(selected_model)
        init_time = time.time() - init_start
        print(f"✅ 客户端初始化完成 (耗时: {init_time:.2f}秒)")
        
        # 分析测试文本
        print("\n📊 分析测试文本...")
        test_text = "Hello world! 你好世界！这是一个测试文本。"
        analysis = client.analyze_text(test_text)
        print(f"   文本: {test_text}")
        print(f"   Token数: {analysis['total_tokens']}")
        print(f"   字符数: {analysis['character_count']}")
        print(f"   中文字符: {analysis['chinese_chars']}")
        print(f"   英文字符: {analysis['english_chars']}")
        
        # 生成答案
        print(f"\n🤖 正在生成回答...")
        print(f"   问题: {test_prompt}")
        generate_start = time.time()
        answer = client.generate_answer(test_prompt, max_tokens=1000)
        generate_time = time.time() - generate_start
        
        # 打印结果
        print("\n" + "=" * 80)
        print("📝 生成结果:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        # 打印性能信息
        print(f"\n⏱️  性能统计:")
        print(f"   初始化耗时: {init_time:.2f}秒")
        print(f"   生成耗时: {generate_time:.2f}秒")
        print(f"   总耗时: {init_time + generate_time:.2f}秒")
        
        # 分析回答
        answer_analysis = client.analyze_text(answer)
        print(f"\n📊 回答分析:")
        print(f"   字符数: {answer_analysis['character_count']}")
        print(f"   Token数: {answer_analysis['total_tokens']}")
        print(f"   中文字符: {answer_analysis['chinese_chars']}")
        print(f"   英文字符: {answer_analysis['english_chars']}")
        
        # 打印模型信息
        info = client.get_model_info()
        print(f"\n🔧 模型信息:")
        print(f"   配置模型: {info['model_name']}")
        print(f"   实际模型: {info['actual_model']}")
        print(f"   提供商: {info['provider']}")
        print(f"   API地址: {info['base_url']}")
        print(f"   上下文长度: {info['context_length']:,}")
        print(f"   最大输出: {info['max_output_tokens']:,}")
        print(f"   默认输出: {info['default_output_tokens']:,}")
        print(f"   支持JSON: {info['supports_json_format']}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)