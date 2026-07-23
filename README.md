**Coding Agent 训练部署全流程实践项目**

- 目前训练的模型为规模最小的deepseek-coder-1.3b-base，第一遍预训练仅使用github样本。后续若需增加样本量或更换更大规模的模型，则将增添其他爬虫数据和CodeParrot数据库的文件。  
- 2020年的mac轻薄本CPU无法支持此规模的预训练，将转移到 Google Colab 的虚拟机上进行。  
  
项目结构：  
data_pipeline/  
├── config/  
│   └── settings.py （+.env） # API keys, 代理配置  
├── crawlers/  
│   ├── github_crawler.py  # github优质库抓取，已完成  
│   ├── stackoverflow_crawler.py  # 待完成  
│   └── docs_crawler.py  # 待完成  
├── utils/  # 未使用  
│   ├── proxy_manager.py   # 代理轮换  
│   └── rate_limiter.py    # 请求限流  
├── requirements.txt         
├── clean_data.py  # 数据清洗：语法检查 + 长度过滤 + PIT替换 + 格式化 + 元数据提取  
├── sample_builder.py  # 样本构建：生成不同类样本（代码注释对、代码补全样本、代码修复样本)+去重过滤  
├── build_tokens.py  # Tokenization：使用deepseek-ai/deepseek-coder-1.3b-base的tokenizer  
├── new_pretrain.py  # 适配个人电脑的代码，环境配置 transformers: 4.41.2，trl: 0.9.4，torch: 2.2.2  
├── pretrain.py  # 用于虚拟机的代码，环境配置均为最新版本  
├── evaluation.py # 检查模型困惑度  
├── check_tokenizer.py # 检查final model中的tokenizer能否正常工作  
├── eval_quick.py # 检查模型的基础代码能力是否正常，由于保存模型时tokenizer存在问题，所以输出格式需要手动调整   
├── sft.py  # 在虚拟机上进行指令微调，使用sample_builder.py中生成的training_sample.json文件   
├── sft_eval.py  # 在虚拟机上评估模型编程能力的代码，有三个问题指令   
├── sft_perplexity.py  # 在虚拟机上计算模型的困惑度，目前困惑度2.65  
├── eval_local.py  # 在个人电脑上评估模型编程能力的代码，dtype = float32   
├── perplexity_local.py  # 在个人电脑上计算模型的困惑度，由于cpu局限性无法得到精准结果   
