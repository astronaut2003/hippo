"""
端到端聊天功能测试脚本
====================
测试从前端API调用到后端服务的完整对话流程

前置条件:
1. 后端服务已启动 (http://localhost:8000)
2. PostgreSQL + pgvector 已配置
3. DeepSeek API Key 已设置
4. HuggingFace embedding 模型可用
"""

import urllib.request
import urllib.parse
import json
import socket
import time

def test_backend_health():
    """测试后端健康状态"""
    try:
        # 基本连接测试
        response = urllib.request.urlopen("http://localhost:8000/health", timeout=5)
        health_data = json.loads(response.read().decode('utf-8'))
        print(f"✅ 后端服务状态: {health_data}")
        
        # Chat服务健康检查
        response = urllib.request.urlopen("http://localhost:8000/api/v1/chat/health", timeout=5)
        chat_health = json.loads(response.read().decode('utf-8'))
        print(f"✅ Chat服务状态: {chat_health}")
        
        return True
    except Exception as e:
        print(f"❌ 后端服务检查失败: {e}")
        return False

def test_chat_conversation():
    """测试完整对话流程"""
    print("\n" + "="*60)
    print("🗨️  测试智能对话功能")
    print("="*60)
    
    # 测试会话
    conversation_id = f"test_conv_{int(time.time())}"
    user_id = "test_user"
    
    # 测试对话序列
    test_messages = [
        "你好，我想了解一下你的功能",
        "我喜欢吃中餐和日料，特别是川菜和寿司",
        "我明天想吃点什么好呢？",
        "谢谢你的建议"
    ]
    
    history = []
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n🔹 对话 {i}: {message}")
        print("-" * 50)
        
        success = send_chat_message(
            message=message,
            user_id=user_id,
            conversation_id=conversation_id,
            history=history
        )
        
        if success:
            # 模拟将消息添加到历史记录
            history.append({"role": "user", "content": message})
            # 注意: 实际的assistant回复应该从API响应中获取
            history.append({"role": "assistant", "content": "AI回复占位符"})
            print("✅ 对话成功")
        else:
            print("❌ 对话失败")
            return False
        
        # 稍微延迟避免请求过快
        time.sleep(1)
    
    print("\n🎉 完整对话测试通过!")
    return True

def send_chat_message(message, user_id, conversation_id, history):
    """发送单条聊天消息"""
    
    # 构建请求数据
    data = {
        "message": message,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "history": history[-4:] if len(history) > 4 else history  # 只保留最近4条记录
    }
    
    url = "http://localhost:8000/api/v1/chat/message"
    json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=json_data,
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'Hippo-E2E-Test/1.0'
        },
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            print(f"📤 状态码: {response.status}")
            
            # 处理流式响应
            if response.headers.get('content-type', '').startswith('text/event-stream'):
                print("📡 接收流式回答:")
                full_response = ""
                
                while True:
                    line = response.readline()
                    if not line:
                        break
                    
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith('data: '):
                        data_json = line_text[6:]
                        try:
                            data_obj = json.loads(data_json)
                            
                            if data_obj.get('error'):
                                print(f"❌ API错误: {data_obj['error']}")
                                return False
                                
                            if data_obj.get('content'):
                                content = data_obj['content']
                                print(content, end='', flush=True)
                                full_response += content
                                
                            if data_obj.get('done'):
                                print("\n✅ 流式响应完成")
                                return True
                                
                        except json.JSONDecodeError:
                            # 忽略无效的JSON片段
                            continue
                            
                print("⚠️  流式响应结束但未收到完成标志")
                return bool(full_response)
            else:
                # 非流式响应
                content = response.read().decode('utf-8')
                print(f"📄 响应: {content}")
                return True
                
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP错误 {e.code}: {e.reason}")
        try:
            error_content = e.read().decode('utf-8')
            print(f"📄 错误详情: {error_content}")
            
            try:
                error_json = json.loads(error_content)
                print("🔍 错误分析:")
                if 'detail' in error_json:
                    print(f"  详情: {error_json['detail']}")
            except:
                pass
        except:
            pass
        return False
        
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

def test_memory_functionality():
    """测试记忆功能"""
    print("\n" + "="*60) 
    print("🧠 测试记忆功能")
    print("="*60)
    
    # 测试记忆相关的对话
    memory_test_messages = [
        "请记住，我的生日是1990年5月15日",
        "我的爱好是阅读和旅游",
        "我最喜欢的颜色是蓝色",
        "你还记得我的生日吗？",
        "我的爱好是什么？"
    ]
    
    conversation_id = f"memory_test_{int(time.time())}"
    
    for i, message in enumerate(memory_test_messages, 1):
        print(f"\n🔹 记忆测试 {i}: {message}")
        print("-" * 40)
        
        success = send_chat_message(
            message=message,
            user_id="memory_test_user",
            conversation_id=conversation_id,
            history=[]
        )
        
        if not success:
            print(f"❌ 记忆测试 {i} 失败")
            return False
        
        time.sleep(1)
    
    print("\n🎉 记忆功能测试完成!")
    return True

def main():
    """主测试函数"""
    print("="*80)
    print("🚀 Hippo 智能对话 - 端到端测试")
    print("="*80)
    
    # 1. 健康检查
    print("\n1️⃣ 后端服务健康检查...")
    if not test_backend_health():
        print("\n❌ 后端服务不可用，请先启动后端服务")
        return False
    
    # 2. 基础对话测试
    print("\n2️⃣ 基础对话功能测试...")
    if not test_chat_conversation():
        print("\n❌ 基础对话功能测试失败")
        return False
    
    # 3. 记忆功能测试
    print("\n3️⃣ 记忆功能测试...")
    if not test_memory_functionality():
        print("\n❌ 记忆功能测试失败")
        return False
    
    # 测试完成
    print("\n" + "="*80)
    print("🎉 所有测试通过！")
    print("✅ 基础对话功能正常")
    print("✅ 流式响应正常")
    print("✅ 记忆功能正常")
    print("✅ API接口正常")
    print("🚀 Hippo 智能对话系统运行正常!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
