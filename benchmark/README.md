# 外部接触角样本验证

本目录负责验证数据的来源、去重、开发/测试分组和阈值统计，不依据标签改变测量方法。用户于 2026-11-16 调整验收阈值，当前默认严格为光滑 `<2°`、轻度噪声 `<3°`、明显噪声 `<5°`，每档目标为 `≥80%`。测量失败保留在分母中；缺失参考值的样本不能进入正式清单。早期报告使用 `<1° / <2.5° / <5°`，旧报告明确保存其阈值。

## 清单约定

`image_path` 相对 manifest 文件所在目录，指向最终用于测量的无角度标注图像。如果原始论文是多面板图，先保存各独立液滴的裁剪图；`raw_sha256` 是该输入图片的精确字节哈希。可用 `source_asset_sha256` 记录整幅论文图的哈希，`crop` 只记录裁剪位置，不会再次裁剪。`baseline` 坐标同样相对最终输入图像。

每个液滴只计一次；同一液滴的相邻视频帧、不同曝光、重复截图共用 `group_id`，不能跨开发集和测试集。原文件哈希和解码后的像素都会去重。`grade_reason` 要先根据接触区的可见边界、模糊、反光、纤维遮挡程度记录，不能根据算法误差倒推分级，也不能将材料微观粗糙程度直接等同于图像噪声。

`reference_kind: sides` 要求来源提供精确对应图像的左右角，严格比较两侧。`reference_kind: mean` 只能比较来源的单图平均角，不会伪造左右标签。两种统计单独分层。实验原图、论文面板和来源发布的合成样例分别标记 `real`、`source-figure`、`synthetic`；不能用增强或合成图片充当新增的真实实验样本。

示例（单条结构演示，正式清单 `expected_count` 应为100）：

```json
{
  "benchmark_id": "external-contact-angle-v1",
  "version": "1",
  "expected_count": 1,
  "grade_policy": "Before scoring, inspect contact-region visibility and document every grade.",
  "grouping_policy": "All images of the same physical drop or experimental sequence stay in one split.",
  "frozen_at": "2026-11-16",
  "samples": [{
    "sample_id": "paper-doi-case001",
    "image_path": "images/case001.png",
    "source_url": "https://example.org/source-dataset",
    "doi": "10.0000/example",
    "source_type": "real",
    "raw_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "grade": "mild",
    "grade_reason": "Mild blur, continuous visible outline and both contact locations remain resolved.",
    "group_id": "paper-doi-physical-drop001",
    "split": "test",
    "reference_kind": "sides",
    "reference_left": 120.1,
    "reference_right": 121.2,
    "reference_uncertainty": null,
    "reference_method": "Published independent manual analysis",
    "reference_locator": "Supplementary table S1, case 001",
    "licence": "CC BY 4.0"
  }]
}
```

无不确定性说明的论文读数可以验证相对来源角度的一致性，但不能据此声称真实准确度达到1°。结果会保留参考不确定性，不会将未知不确定性当作零。

## 运行

在项目目录安装测量依赖后执行：

```bash
python -m benchmark.runner /absolute/path/manifest.json \
  --adapter benchmark.adapters:fibrous \
  --before-adapter benchmark.adapters:original \
  --output results/external-benchmark
```

默认 `benchmark.adapters:current` 在提供基线时使用原核心，没有基线时使用纤维流水线。`fibrous` 总是使用纤维流水线；`original` 要求提供基线，缺失时计为失败。其他适配器用 `module:function` 指定，签名为 `callback(image_bytes, baseline=None)`，必须返回包含角度的字典或原始 `MeasurementOutput`。适配器收到图片和基线，不会收到来源编号、分级、参考角度或阈值。

结果包括 `report.json`、`records.csv`、`summary.json`，有前后适配器时额外生成配对比较。每条失败及其原因均保留。清单的 SHA256 指纹确保前后使用同一套样本与标签。汇总按噪声等级、开发/测试集、来源类型、参考形式分层；缺少某档样本时，该档不可能满足总目标。

用户更改验收标准时，可调用 `rescore_report(old_report)` 按当前阈值对缓存误差重新计分，无需重新拟合。它保留全部预测和失败，记录原报告指纹及旧阈值，不修改输入报告。`compare` 要求前后清单与阈值均一致，避免将阈值放宽误报为算法改进。

运行本目录的验证：

```bash
python -m pytest benchmark/tests -q
```

这套工具不替代独立冻结验收。对测试集结果反复调参之后，该测试集应改记为开发集，并另行冻结未查看的新测试集。

### 论文只报告一个角度而侧别规则未知

使用 `reference_kind: "reported_scalar"` 和 `reference_scalar`，让 `reference_left`、`reference_right`、`reference_mean` 保持 null。适配器的图像级算术均值将与这个来源标量比较，但结果明确标记为：

```text
image_level_against_published_scalar_unknown_side_convention
```

它不能被描述为已知左右均值，也不能证明两侧都通过。`error_reported_scalar` 单列误差；`score_scope` 在测量失败时仍保留；汇总输出 `reference_kind_counts` 和按参考形式分层的结果，未知参考不确定性也单独计数。此类样本只能支持“与论文所报图像级标量的一致性”，不能支持双侧准确度或明确的平均角准确度。
