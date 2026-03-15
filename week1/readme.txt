MALearn / LangChain 7天入门学习说明

一、项目定位
本项目用于配合 LangChain / LangGraph 入门教学，目标不是一开始就做复杂系统，而是按照“从简单到完整”的顺序，带着学习者在 7 天内逐步完成以下能力：
1. 连接本地大模型服务；
2. 理解工具调用（tool calling）的基本思想；
3. 学习 LangGraph 的状态流、节点和条件边；
4. 理解最小多智能体的角色分工；
5. 组合成科研学习助手原型；
6. 加入反思与重试机制；
7. 将固定知识升级为本地知识库检索。

本项目适合以下学习目标：
- 初学 LangChain / LangGraph；
- 想理解 agent 系统的基本组成；
- 想从“能调用模型”逐步走到“能构建一个可解释的小型智能体系统”；
- 想为后续学习 RAG、多智能体协作、科研学习助手打基础。


二、项目环境说明
本项目默认使用本地 vLLM 服务，配合 LangChain 的 OpenAI 兼容接口调用。

1. Python 环境启动方式
source /workspace/codes/python_venv_new/bin/activate

2. 本地模型服务示例（Qwen3）
vllm serve "/workspace/codes/LLM/Qwen3-4B" \
  --served-model-name local-qwen3 \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000

3. 依赖安装
建议先进入虚拟环境，再安装依赖：
pip install langchain langgraph langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple

4. 本项目配置风格
为了便于教学和后续维护，项目采用如下组织方式：
- config.py：存放模型配置、路径、提示词、常量；
- utils.py：存放通用函数，例如 get_llm、输出清洗、文本处理等；
- tools.py：存放工具能力，例如论文知识工具、代码帮助工具、数学计算器等；
- 各 Day 脚本：存放当天的主流程。


三、建议目录结构
week1/
├── data/
│   └── agent_notes.txt
└── demos/
    ├── config.py
    ├── utils.py
    ├── tools.py
    ├── 01_chat_with_vllm.py
    ├── 02_tool_calling_demo.py
    ├── 03_langgraph_router.py
    ├── 04_multi_agent_demo.py
    ├── 05_research_assistant_demo.py
    ├── 06_reflection_retry_demo.py
    └── 07_local_kb_assistant.py


四、公共模块说明

（一）config.py
作用：
1. 集中管理本地 vLLM 配置；
2. 集中管理系统提示词；
3. 集中管理知识库路径等常量。

学习重点：
- 为什么要把配置集中管理；
- 为什么提示词不应该散落在每个脚本里；
- 为什么后续做 agent 系统时，config.py 会越来越重要。


（二）utils.py
作用：
1. 提供 get_llm()，统一获取 LLM 实例；
2. 提供 clean_model_output()，清理 <think> 等输出痕迹；
3. 提供 print_section_line() 等通用输出函数；
4. 后续还可以补充文本处理、表达式判断等通用逻辑。

学习重点：
- 为什么 build_llm 和 get_llm 要写成公共函数；
- 为什么输出清洗应放在 utils.py，而不是散落在各个脚本里；
- 通用函数如何帮助主脚本保持简洁。


（三）tools.py
作用：
1. 提供论文知识工具，例如 ReAct、Toolformer、Reflexion；
2. 提供代码帮助工具，例如 think_clean、python_function、debug；
3. 后续可继续扩展数学计算器、本地检索器等外部能力。

学习重点：
- tool 的本质是“系统可调用的外部能力”；
- tool 与 LLM 的区别：LLM 负责组织语言，tool 负责提供外部信息或外部计算结果；
- 为什么 agent 系统通常需要 tools。


五、7天学习路线

====================
Day1：连接本地大模型
====================
文件：01_chat_with_vllm.py

学习目标：
1. 跑通 LangChain 对本地 vLLM 的调用；
2. 理解 ChatOpenAI + base_url 的使用方式；
3. 理解“本地模型服务”和“Python 客户端对象”的区别。

脚本做了什么：
- 通过 utils.py 中的 get_llm()/build_llm() 创建模型调用对象；
- 构造一个中文问题；
- 调用 llm.invoke() 输出模型结果。

运行方式：
cd /workspace/codes/MALearn/week1/demos
python 01_chat_with_vllm.py

建议观察：
1. 是否能正常连接本地服务；
2. 是否稳定使用中文回答；
3. 模型服务名是否与 config.py 中一致。

学习收获：
完成 Day1 后，应理解：
- LangChain 并不是“另一个模型”，而是模型调用和流程编排框架；
- 本地 vLLM 通过 OpenAI 兼容接口即可接入 LangChain。


====================
Day2：工具增强回答
====================
文件：02_tool_calling_demo.py

学习目标：
1. 理解 tool calling 的最小形态；
2. 理解“先用工具拿信息，再让模型组织回答”；
3. 理解系统提示词和输出清洗的作用。

脚本做了什么：
- 接收用户问题；
- 调用 tools.py 中的 search_paper_topic() 等工具获取知识；
- 用模型基于工具结果回答问题；
- 用 utils.py 中的 clean_model_output() 清洗输出。

运行方式：
cd /workspace/codes/MALearn/week1/demos
python 02_tool_calling_demo.py

建议观察：
1. 工具返回的内容是否真正进入模型回答；
2. 模型是否主要围绕工具结果回答；
3. <think> 等痕迹是否被正确清洗。

学习收获：
完成 Day2 后，初学者应理解：
- tool 不等于复杂 agent，它最开始只是一个 Python 函数；
- 真正重要的是“信息流”：用户问题 -> tool -> LLM -> 回答。


====================
Day3：LangGraph 入门
====================
文件：03_langgraph_router.py

学习目标：
1. 理解 LangGraph 的 State、Node、Edge；
2. 理解 add_conditional_edges 的用法；
3. 理解“节点负责更新状态，边负责决定流向”。

脚本做了什么：
- 定义共享状态 AgentState；
- 定义 router_node、research_node、coder_node、answer_node；
- 使用条件边根据 state 中的 route 选择分支；
- 最后统一由 answer_node 输出答案。

运行方式：
cd /workspace/codes/MALearn/week1/demos
python 03_langgraph_router.py

建议观察：
1. state 中的 route、topic、tool_result 是何时被写入的；
2. 为什么要单独写 route_edge()；
3. 为什么 answer_node 不直接写在 research/coder 节点内部。

学习收获：
完成 Day3 后，初学者应理解：
- LangGraph 不只是“多写几个函数”，而是把流程显式表示为图；
- State 是共享记录表，Node 负责写状态，Edge 负责调度。


====================
Day4：最小多智能体
====================
文件：04_multi_agent_demo.py

学习目标：
1. 理解 supervisor 与 specialist 的最小协作模式；
2. 理解 research_agent / coder_agent 的角色分工；
3. 理解多智能体不是越多越好，而是职责清楚最重要。

脚本做了什么：
- 使用 supervisor_route_node 先分配任务；
- 使用 research_agent_node / coder_agent_node 生成专业中间结果；
- 使用 supervisor_finalize_node 汇总最终回答；
- 使用 get_llm() 复用 LLM 对象。

运行方式：
cd /workspace/codes/MALearn/week1/demos
python 04_multi_agent_demo.py

建议观察：
1. specialist_result 与 final_answer 的区别；
2. 为什么要统一汇总，而不是让每个 agent 直接输出终稿；
3. 代码类任务与研究类任务在中间结果上的差异。

学习收获：
完成 Day4 后，初学者应理解：
- 多智能体的关键是分工，而不是多个模型同时说话；
- supervisor 负责调度和汇总，specialist 负责提供专业结果。


====================
Day5：科研学习助手原型
====================
文件：05_research_assistant_demo.py

学习目标：
1. 把前几天的能力组合成一个更完整的应用型 demo；
2. 理解复合任务拆解；
3. 理解一个任务可能同时需要 research 和 coder 两种能力。

脚本做了什么：
- 用 planner_node 判断任务是否需要 research / coder；
- 用 executor_node 按需调用 research_agent 和 coder_agent；
- 用 finalizer_node 整合 research_result 和 code_result；
- 形成最小科研学习助手原型。

运行方式：
cd /workspace/codes/MALearn/week1/demos
python 05_research_assistant_demo.py

建议观察：
1. need_research / need_coder 是否判断合理；
2. 复合问题是否同时触发两类中间结果；
3. final_answer 是否真正整合了两类结果。

学习收获：
完成 Day5 后，初学者应理解：
- 一个任务可以拆成多个子需求；
- 不同 specialist 可以为同一个最终答案共同服务。


====================
Day6：反思与重试
====================
文件：06_reflection_retry_demo.py

学习目标：
1. 理解最小 reflection + retry 闭环；
2. 理解 checker 的作用；
3. 理解为什么 agent 系统不应只生成一次就结束。

脚本做了什么：
- 使用 first_finalizer_node 先生成第一版答案；
- 使用 checker_node 检查草稿痕迹、代码信号、数学结果等；
- 若检查失败，则通过 retry_finalizer_node 根据 reflection_note 重写答案；
- 若通过，则直接 pass_through。

运行方式：
cd /workspace/codes/MALearn/week1/demos
python 06_reflection_retry_demo.py

建议观察：
1. 哪些任务会触发 retry；
2. reflection_note 是如何描述问题的；
3. retry 后的答案是否更稳、更干净。

学习收获：
完成 Day6 后，初学者应理解：
- agent 系统不一定一次就给出最佳答案；
- 生成、检查、重试是一个很重要的质量控制闭环；
- checker 可以是规则型，也可以后续扩展成模型型。

注意：
当前数学检查更适合显式表达式题，例如 37 × 48 + 125，不适合直接处理鸡兔同笼这类文字应用题。


====================
Day7：本地知识库检索
====================
文件：07_local_kb_assistant.py
数据文件：data/agent_notes.txt

学习目标：
1. 理解“知识从 tools.py 固定字典迁移到本地知识文件”的意义；
2. 理解最小检索增强流程；
3. 为后续学习真正 RAG 做准备。

脚本做了什么：
- 从 agent_notes.txt 读取知识；
- 将文本按空行切块；
- 按关键词匹配检索 top-k chunk；
- 用检索片段生成第一版答案；
- 若答案有问题，再执行 retry。

运行方式：
cd /workspace/codes/MALearn/week1/demos
python 07_local_kb_assistant.py

建议观察：
1. retrieved_chunk_ids 是否合理；
2. retrieved_chunks 是否和用户问题对应；
3. first_answer 是否明显基于检索到的知识片段；
4. retry 是否在必要时触发。

学习收获：
完成 Day7 后，初学者应理解：
- 外部知识可以不写死在代码中，而是从本地文件中检索；
- 检索增强生成（RAG）的最小思想已经具备：先检索，再生成。


六、推荐学习顺序
1. 先跑 Day1，确认本地模型调用正常；
2. 再跑 Day2，理解工具增强；
3. 用 Day3 学会 LangGraph 的状态流与条件边；
4. 用 Day4 学会 supervisor-specialist 的最小多智能体模式；
5. 用 Day5 理解复合任务拆解；
6. 用 Day6 理解反思与重试；
7. 最后用 Day7 理解本地知识库检索。

不建议跳着学，因为这 7 天内容是逐步递进的：
- 如果没理解 Day2 的 tool，就很难理解 Day5 的 executor；
- 如果没理解 Day3 的 state 和 conditional edge，就很难真正看懂 Day4 的多智能体流；
- 如果没理解 Day6 的 checker / retry，就很难在 Day7 里理解检索回答后的质量控制。


七、给初学者的学习建议
1. 不要只看最终输出，要重点看中间状态；
2. 不要只会运行代码，要学会回答“为什么这样设计”；
3. 每天都建议先读主脚本，再读 config.py / utils.py / tools.py；
4. 建议自己修改 demo_cases，多做输入实验；
5. 学到 Day7 后，可以开始思考：如何从关键词检索升级成向量检索。


八、后续可继续扩展的方向
1. 将 Day7 的关键词检索升级成向量检索；
2. 给知识库加入更多文档；
3. 增强 checker，使其更严格判断 groundedness；
4. 将 Day5 的科研学习助手与 Day7 的本地检索结合；
5. 进一步学习 LangGraph 的循环、多分支和更复杂的多智能体模式。


九、结语
这套项目的教学目标，不是让初学者一次性掌握所有高级 agent 技术，而是通过 7 天的递进学习，建立起对以下主线的完整理解：

本地模型调用
-> 工具增强
-> 状态流与路由
-> 最小多智能体
-> 科研学习助手
-> 反思与重试
-> 本地知识库检索

只要按顺序完成并理解每一天的脚本，初学者就已经具备继续学习更复杂 LangChain / LangGraph / RAG 系统的基础。
