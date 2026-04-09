
#实现接口LLM
from dotenv import load_dotenv
import os
from typing import Dict
from typing import List
import openai
from openai import OpenAI
from asyncio.windows_events import NULL
from serpapi import SerpApiClient

# 加载环境变量
load_dotenv()
# 从环境变量中获取配置
api_key_load = os.getenv("OPENAI_API_KEY")
base_url_load = os.getenv("OPENAI_BASE_URL")
model_load = os.getenv("LLM_MODEL_ID")
class LLM:
    # 实现接口openai
    def __init__(self, model: str = None, base_url: str = None, api_key: str = None):
        self.model = model or model_load
        self.base_url = base_url or base_url_load
        self.api_key = api_key or api_key_load
        self.client = OpenAI( base_url=self.base_url, api_key=self.api_key)

    def think(self,messages: List[Dict[str, str]]):
        if(self.client == None):
            raise ValueError("client is empty")
        # 调用openai api 获取回复
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content



#提示词
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。
可用工具如下：
 {tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- {{tool_name}}[{{tool_input}}]：调用一个可用工具。
- Finish[最终答案]：当你认为已经获得最终答案时。

现在，请开始解决以下问题：
Question: {question}
History: {history}
"""



#使用函数代替工具类
def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智能解析：优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"

def add(a: int, b: int) -> int:
    """
    一个简单的加法工具，用于计算两个整数的和。
    """
    return a + b


# 实现ReActAgent
class ReActAgent:
    def __init__(self, llm: LLM, tools: List[Dict[str, str]], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []
        self.current_step = 0
    def run(self, question: str):
        self.history = []
        self.current_step = 0
        while self.current_step < self.max_steps:
            self.current_step += 1
            print(f"\n--- 第 {self.current_step} 步 ---")
            tools_desc = "\n".join([f"{tool['name']}: {tool['description']}" for tool in self.tools])
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。"); break
            self.history.append(response_text)
            print(f"LLM 回复: {response_text}")
            
            # 解析LLM回复
            if "Finish[" in response_text:
                final_answer = response_text.split("Finish[")[1].split("]")[0]
                print(f"✅ 已完成任务，最终答案: {final_answer}")
                return final_answer
            
            if "Action[" in response_text:
                action = response_text.split("Action[")[1].split("]")[0]
                tool_name, tool_input = action.split("[")
                tool_input = tool_input.strip()
                
                # 调用对应工具
                if tool_name == "search":
                    result = search(tool_input)
                elif tool_name == "add":
                    try:
                        a, b = map(int, tool_input.split(","))
                        result = add(a, b)
                    except ValueError:
                        result = "错误：add 工具需要两个整数参数，格式为 'a,b'"
                else:
                    result = f"错误：未知工具 '{tool_name}'"
                    break
                
                # 将工具执行结果追加到历史记录中，供 LLM 下一步思考使用
                print(f"工具执行结果: {result}")
                self.history.append(f"Observation: {result}")

if __name__ == "__main__":
    llm = LLM()
    tools = [{"name": "search", "description": "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"}, {"name": "add", "description": "用于计算两个整数的和"}]
    agent = ReActAgent(llm, tools)
    question = "最新款的奔驰e300系列有哪些？它的主要卖点是什么？"
    agent.run(question)
    
