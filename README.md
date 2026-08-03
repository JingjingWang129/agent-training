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
├── processing_data/   
│   ├── clean_data.py  # 数据清洗：语法检查 + 长度过滤 + PIT替换 + 格式化 + 元数据提取   
│   ├── sample_builder.py  # 样本构建：生成不同类样本（代码注释对、代码补全样本、代码修复样本)+去重过滤   
│   ├── build_tokens.py  # Tokenization：使用deepseek-ai/deepseek-coder-1.3b-base的tokenizer     
│   └──  select_sft_samples.py  # 从原本的training dataset中筛选出更为优质的1000条样本用于新sft   
├── trying_failed/   
│   ├── new_pretrain.py  # 适配个人电脑的代码，环境配置 transformers: 4.41.2，trl: 0.9.4，torch: 2.2.2   
│   ├── pretrain.py  # 用于虚拟机的代码，环境配置 transformers: 5.14.1，trl: 1.8.0，torch: 2.8.0    
│   ├── evaluation.py # 检查模型困惑度   
│   ├── eval_quick.py # 检查模型的基础代码能力是否正常，由于保存模型时tokenizer存在问题，所以输出格式需要手动调整  
│   ├── sft.py  # 在虚拟机上进行指令微调，使用sample_builder.py中生成的training_sample.json文件，共22万余样本    
│   ├── sft_eval.py  # 在虚拟机上评估模型编程能力的代码，有三个自然语言问题指令，模型的回复没有分词分行     
│   ├── sft_perplexity.py  # 在虚拟机上计算模型的困惑度，目前困惑度2.65   
│   ├── eval_local.py  # 在个人电脑上评估模型编程能力的代码，dtype = float32   
│   ├── perplexity_local.py  # 在个人电脑上计算模型的困惑度，由于cpu局限性无法得到精准结果    
│   └── sft_v2.py  # 基于第一次sft模型基础上的第二次sft，模型困惑度2.85，输出格式问题依然存在   
├── debug_tokenizer.py  # 检查不同阶段模型的tokenizer能否正确转译training dataset中的代码格式   
├── test_tokenizer.py  # 在确认tokenizer存在问题后逐步排查问题成因，最终确定是其与transformers v5的AutoTokenizer不兼容导致的  
├── sft_v3.py  # 基于deepseek-coder-1.3b-base的原始模型进行新一轮轻量化sft，样本数1000   
├── new_eval.py  # 对新sft模型进行编程功能评估，模型困惑度2.25，在添加重复惩罚后输出格式符合需要   
└── 模型输出总览.rtf  # 最终版sft模型对于不同难度的编程问题的回答一览   
