import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from tools import search  # 假设 tools.py 中有你之前写的 search 方法

# 加载环境变量
load_dotenv()

# 初始化 LLM
# 这里使用 langchain_openai，它会自动从环境变量读取 OPENAI_API_KEY 和 OPENAI_BASE_URL
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "qwen-plus"),
    temperature=0.7,
)

# 使用 @tool 装饰器将现有的 search 函数包装成 LangChain 支持的工具
@tool
def search_tool(query: str) -> str:
    """一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。输入应该是具体的搜索关键词。"""
    return search(query)

# 定义工具列表
tools = [search_tool]

# 定义系统提示词
# 虽然 LangGraph 的 create_react_agent 不强制需要 prompt，
# 但通过 SystemMessage 提供明确的系统设定可以让 Agent 表现得更聪明、更符合预期
system_prompt = SystemMessage(content="""
你是一个有能力调用外部工具的智能助手。
当用户提出问题时，如果你不知道答案，你应该主动使用 Search 工具去搜索最新的网页信息。
搜索到信息后，请进行总结和归纳，最后用中文给出准确、清晰的回答。
""")

# 在 LangChain 最新版 (基于 LangGraph) 中，创建 ReAct Agent 非常简单
# 不需要手写复杂的 prompt 模板，也不需要 AgentExecutor
# create_react_agent 会返回一个编译好的图 (CompiledGraph)
agent = create_react_agent(
    model=llm, 
    tools=tools,
    state_modifier=system_prompt  # 使用 state_modifier 传入系统提示词
)

if __name__ == "__main__":
    question = "最新款的奔驰e300系列有哪些？它的主要卖点是什么？"
    print(f"正在解答问题: {question}\n")
    
    # 构造输入消息
    inputs = {"messages": [("user", question)]}
    
    # 执行 Agent (使用 stream 可以观察每一步的执行过程)
    print("--- 执行过程 ---")
    for event in agent.stream(inputs, stream_mode="values"):
        # 获取最新的一条消息
        message = event["messages"][-1]
        message.pretty_print()
        
    # 获取最终结果
    final_response = event["messages"][-1].content
    print("\n🎉 最终答案:")
    print(final_response)
