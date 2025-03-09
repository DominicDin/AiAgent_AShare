import os
import time
import requests  # 新增：用于 DeepSeek HTTP 请求
import google.generativeai as genai
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

# 设置日志记录
logger = setup_logger('api_calls')

@dataclass
class ChatMessage:
    content: str

@dataclass
class ChatChoice:
    message: ChatMessage

@dataclass
class ChatCompletion:
    choices: list[ChatChoice]

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 验证环境变量
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")
if not api_key:
    logger.error(f"{ERROR_ICON} 未找到 GEMINI_API_KEY 环境变量")
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not model:
    model = "gemini-1.5-flash"
    logger.info(f"{WAIT_ICON} 使用默认模型: {model}")

# 初始化 Gemini 客户端
genai.configure(api_key=api_key)
client = genai
logger.info(f"{SUCCESS_ICON} Gemini 客户端初始化成功")

# 新增：DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.warning(f"{ERROR_ICON} 未找到 DEEPSEEK_API_KEY 环境变量，DeepSeek 备用方案可能无法使用")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"  # 假设 DeepSeek 的端点为此

def call_deepseek(prompt: str, model: str = "deepseek-chat") -> str:
    """
    调用 DeepSeek API 生成文本
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        logger.info(f"{WAIT_ICON} 正在调用 DeepSeek API...")
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            logger.info(f"{SUCCESS_ICON} DeepSeek API 调用成功")
            return answer
        else:
            logger.error(f"DeepSeek API 返回错误: {resp.status_code}, {resp.text}")
            raise RuntimeError(f"DeepSeek API error: {resp.status_code}")
    except Exception as e:
        logger.error(f"调用 DeepSeek API 异常: {e}")
        raise e

@backoff.on_exception(
    backoff.expo,
    (Exception,),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "AFC is enabled" not in str(e)
)
def generate_content_with_retry(model, contents, config=None):
    """带重试机制的内容生成函数（调用 Gemini API）"""
    try:
        logger.info(f"{WAIT_ICON} 正在调用 Gemini API...")
        logger.debug(f"请求内容: {contents}")
        logger.debug(f"请求配置: {config}")

        if config is None:
            config = {}

        # 调用新版 Gemini API：直接传入 prompt
        response = genai.generate_text(
            model=model,
            prompt=contents,
            **config
        )
        logger.info(f"{SUCCESS_ICON} Gemini API 调用成功")
        logger.debug(f"响应内容: {response.text[:500]}...")
        return response
    except Exception as e:
        if "AFC is enabled" in str(e):
            logger.warning(f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {str(e)}")
            time.sleep(5)
            raise e
        logger.error(f"{ERROR_ICON} Gemini API 调用失败: {str(e)}")
        logger.error(f"错误详情: {str(e)}")
        raise e

def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    """
    获取聊天完成结果，调用顺序为：
    1. DeepSeek API
    2. Gemini API
    只要有一种调用成功，则返回结果，不再调用后续接口。
    """
    try:
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        # 检查并添加前缀（仅 Gemini 需要此处理）
        if not model.startswith("models/") and not model.startswith("tunedModels/"):
            model = "models/" + model

        logger.info(f"{WAIT_ICON} 使用模型: {model}")
        logger.debug(f"消息内容: {messages}")

        # 构造 prompt（将 system 指令合并到 prompt 中）
        prompt = ""
        system_instruction = None
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        if system_instruction:
            prompt = f"System: {system_instruction}\n" + prompt

        logger.debug(f"最终请求 prompt: {prompt.strip()}")

        # 1. 尝试调用 DeepSeek API
        for attempt in range(max_retries):
            try:
                answer = call_deepseek(prompt.strip(), model="deepseek-chat")
                if answer:
                    logger.info(f"{SUCCESS_ICON} DeepSeek API 返回结果")
                    return answer
            except Exception as e:
                logger.error(f"{ERROR_ICON} DeepSeek API 尝试 {attempt+1}/{max_retries} 失败: {e}")
                time.sleep(initial_retry_delay * (2 ** attempt))

        # 2. 如果 DeepSeek 调用失败，则尝试调用 Gemini API
        for attempt in range(max_retries):
            try:
                response = generate_content_with_retry(
                    model=model,
                    contents=prompt.strip(),
                    config={}
                )
                if response is not None:
                    logger.info(f"{SUCCESS_ICON} Gemini API 返回结果")
                    return response.text
            except Exception as e:
                logger.error(f"{ERROR_ICON} Gemini API 尝试 {attempt+1}/{max_retries} 失败: {e}")
                time.sleep(initial_retry_delay * (2 ** attempt))
        # 如果所有 API 均失败，则返回 None
        return None
    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None


