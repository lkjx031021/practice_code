# 我需要一个数据分析对话Agent的前端页面，请使用React + TypeScript + Ant Design + ECharts实现。

# 核心功能：

文件上传区：支持拖拽上传csv/xlsx，

对话界面：类似ChatGPT的聊天布局，AI回复可包含文字和图表

图表渲染：根据后端返回的chart_data自动渲染柱状图、折线图、饼图等

历史记录：左侧保存对话历史，刷新不丢失

# 接口约定：
在现有前端代码上新增页面，尽量不要动原有的页面
上传API：POST /api/upload

对话API：POST /api/chat，请求体格式为{file_id?, question}

# UI要求：

UI样式参考现在的前端代码，

响应式设计

加载状态动画

请给出完整代码，包含组件拆分、状态管理和样式。

# 其他要求
完成后端文件上传处理的代码