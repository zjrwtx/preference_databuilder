<a name="25ef5371"></a>
# 概述：
大模型RLHF（ppo奖励模型）训练偏好数据排序助手（支持ollama本地模型）

简单来说就是你经过sft微调后，想通过RLHF（ppo奖励模型）训练怎么样的模型，就给你的模型生成回答进行排序，最后再导出偏好数据去训练奖励模型，再用奖励模型去训练sft模型

<a name="778d2597"></a>
# 演示视频地址

[大模型RLHF（ppo奖励模型）训练偏好数据排序助手（ollama本地模型版）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1P1421675z/?spm_id_from=333.999.0.0)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/22859856/1714814701177-4ab729b7-81c7-4c72-a878-6ead50141c06.png#averageHue=%23efead7&clientId=u2224f8ff-e3ea-4&from=paste&height=539&id=uda14b597&originHeight=809&originWidth=1722&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=447245&status=done&style=none&taskId=ue2748e40-1357-46f6-b231-b253198bccc&title=&width=1148)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/22859856/1714814858973-a448dee4-7bc6-46b3-af2b-955a26937dd2.png#averageHue=%23fee176&clientId=u2224f8ff-e3ea-4&from=paste&height=598&id=ua7fb792b&originHeight=897&originWidth=1908&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=493572&status=done&style=none&taskId=u540e463a-0125-4d89-9621-f915e3afd36&title=&width=1272)


本项目遵循GPL许可证，欢迎贡献代码或提出改进建议。项目地址：<br />[https://github.com/zjrwtx/preference_databuilder](https://github.com/zjrwtx/preference_databuilder)

<a name="0cfeb4d9"></a>
# 如何运行

1、克隆到本地

```git
git clone https://github.com/zjrwtx/preference_databuilder.git
```

2、安装依赖

```git
poetry install
```

3、配置ollama环境与模型或云端模型

4、复制.env.example文件为.env 填写大模型的环境变量等

5、streamlit run main.py

<a name="bb966aa6"></a>
# 贡献

欢迎贡献。请先 fork 仓库，然后提交一个 pull request 包含你的更改。

<a name="e40a454f"></a>
# 联系我

<a name="da671a4d"></a>
## 微信：

agi_isallyouneed

<a name="e8c53647"></a>
## 微信公众号：正经人王同学

![](https://cdn.nlark.com/yuque/0/2024/jpeg/22859856/1713801561819-9d19cb9a-1233-4295-ad90-56042bbabd3c.jpeg#averageHue=%23a2a1a0&clientId=u7b5f5d88-e731-4&from=paste&height=172&id=u329dbc86&originHeight=430&originWidth=430&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=40862&status=done&style=none&taskId=u7551bc0b-a19a-4ff7-8b6e-1c0d27b3ae1&title=&width=171.66668701171875#averageHue=%23a2a1a0&id=SjL3U&originHeight=430&originWidth=430&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#averageHue=%23a2a1a0&id=dJonX&originHeight=430&originWidth=430&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#averageHue=%23a2a1a0&id=Wxfkz&originHeight=430&originWidth=430&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#averageHue=%23a2a1a0&id=FOaFI&originHeight=430&originWidth=430&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<a name="58082d81"></a>
## X（推特)正经人王同学:[https://twitter.com/zjrwtx](https://twitter.com/zjrwtx)

<a name="20a28457"></a>
# 许可证

本项目遵循GPL许可证，欢迎贡献代码或提出改进建议。项目地址：[https://github.com/zjrwtx/preference_databuilder](https://github.com/zjrwtx/preference_databuilder)

非商业用途：本项目的所有源代码和相关文档仅限于非商业用途。任何商业用途均被严格禁止。

出处声明：任何个人或实体在修改、分发或使用本项目时，必须清楚地标明本项目的原始来源，并且保留原始作者的版权声明。

<a name="0e7685a8"></a>
# 特别感谢
代码参考：<br />[https://github.com/HarderThenHarder/transformers_tasks](https://github.com/HarderThenHarder/transformers_tasks)
