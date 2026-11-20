# 接触角工作台 v0.5.2

个人使用请双击 **启动接触角.command**：它会启动完整本机前端与 Python 后端，并打开浏览器。支持自动轮廓拟合、多点表面基线拟合、统一液滴轮廓打点、受限缩放拖动、精确坐标编辑、历史记录，以及绘有接触点、切线和角度弧线的 PNG 导出。详细操作见 [个人使用说明](README_STUDIO.md)，Codex/浏览器代理操作见 [AGENT_GUIDE.md](AGENT_GUIDE.md)。无需前端构建工具或在线模型。

工作台可直接处理原始方向的倒置液滴。程序识别基体下缘，估计被液滴挡住的表面，在左右接触点分别计算液相内角。结果和点位始终使用上传照片的左右顺序。

下文保留原核心 Python、CLI 与 FastAPI 接口说明；新版工作台的自动与手动流程使用独立的 `/api/` 接口，原 `/measure` 接口继续可用。

本目录保留 2026-11-16 的外部 100 图验证：`general.py`、`young_laplace.py` 和 `benchmark/`。工作台提供这些拟合方法供个人测量和核查，外部验证尚未达到三档各 ≥80% 的要求；它们未替换原命令行与 `/measure` 接口的默认方法。具体调用和限制见文末。

## 当前测量范围

- 固体上方的单个静态座滴，图片中液滴与背景有可识别的明暗边界。
- 手动指定基线；坐标以原图左上角为原点，x 向右、y 向下，单位为像素。
- 左右角均为**液体内部的夹角**，支持锐角、钝角与倾斜基线（±45°内）。
- 默认分别拟合左右接触区的二次参数曲线；可选三次参数曲线或全轮廓圆拟合。
- ROI 选择液滴，excluded_regions 指出不应参与拟合的遮挡/纤维区域。
- PNG、JPEG、单帧 TIFF、BMP、WebP；8 位灰度/RGB；上限 20 MiB、1200 万像素。

当前参考测试覆盖 30°—150° 的合成圆弧、倾斜基线及已知端点切线的不对称曲线。真实实验图像的精度仍需用参考测量校验。程序允许的 5°—175° 是计算边界，不是已验证精度范围。

## 给 Codex 使用

将图片交给 Codex 后，可按下列流程运行：

1. 查看原图，确定液滴、固体表面和接触区；记录基线端点和 ROI。
2. 若有可辨认的纤维/遮挡，记录排除区域。真实接触点完全不可见时应说明证据不足。
3. 保存 options.json，调用命令行或 `measure_bytes`。
4. 查看 annotated.png 和 sensitivity.csv，核对识别轮廓、基线和失败原因。
5. 报告左右主测量值、通过检查的方法的中位数/范围、方法数量和需要复核的因素。

左右角明显不一致时分别报告。只有能解释图像差异时才调整 ROI/排除区域，不按期望角度挑选参数或方法。

## 安装

建议 Python 3.11 或更高版本。当前实际测试环境为 macOS Apple Silicon / Python 3.13。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

`requirements-lock.txt` 记录本次验证的精确依赖版本（包含测试工具）；其他平台若缺少相应 wheel，可用 `pip install -e '.[dev]'` 重新解析兼容依赖，并运行测试。普通用户安装包尚未制作。

## 一次命令完成测量

在项目目录中执行：

```bash
contact-angle measure examples/demo_90.png \
  --options examples/demo_90.options.json \
  --output results/demo_90
```

生成：

| 文件 | 内容 |
|---|---|
| `result.json` | 全部参数、左右角、拟合残差、轮廓坐标、逐方法结果、图片 SHA256 和算法版本 |
| `results.csv` | 主测量值与敏感性摘要，Excel 可打开 |
| `sensitivity.csv` | 每个候选的方法、参数、角度、接受/排除/失败状态和原因 |
| `annotated.png` | 基线（蓝）、轮廓（绿）、接触点和切线（红）、排除区域（紫） |

默认不覆盖已有结果；确需替换时使用 `--overwrite`。原图不修改。关闭敏感性分析时 sensitivity.csv 只含表头。

## 参数示例

```json
{
  "baseline": [{"x": 120, "y": 600}, {"x": 780, "y": 600}],
  "method": "local_polynomial",
  "polarity": "dark",
  "polynomial_degree": 2,
  "fit_fraction": 0.12,
  "blur_sigma": 0.8,
  "threshold": null,
  "baseline_margin_px": 1.5,
  "sensitivity": true,
  "sensitivity_alert_deg": 3.0,
  "excluded_regions": []
}
```

- `baseline`：必填，两个端点顺序可以互换；定义一条直线，不会限制液滴宽度。端点均需位于图内。
- `roi`：可选，例如 `{"x":200,"y":350,"width":500,"height":280}`。坐标始终相对原图。应保留完整液滴及接触区，不把液滴裁断。
- `polarity`：`dark` 为深色液滴/浅色背景，`light` 反之。
- `threshold: null`：根据基线上方的 ROI 图像自动用 Otsu 阈值；也可固定 0—255 阈值。
- `fit_fraction`：每一侧使用的轮廓弧长占总开放轮廓弧长的比例，0.04—0.30。
- `polynomial_degree`：2 或 3。参数曲线分别表示 x(s)、y(s)，可处理超过 90° 的接触角。
- `baseline_margin_px`：排除与基线紧邻的像素带，减少基线截断对拟合的影响；保持固定而不在敏感性比较中移动。
- `excluded_regions`：矩形列表，格式与 ROI 相同；被覆盖的轮廓点不进入拟合。排除区域不用于生成/修补真实轮廓；只在计算弧长坐标时跨过空隙，外推过远会拒绝测量。
- `sensitivity: false`：只运行指定方法。

图片不要做横纵比例不同的缩放；无需物理长度标定即可计算角度，但像素坐标应具有相同的横纵尺度。带未归一化 EXIF 方向的照片会要求先统一方向，避免基线坐标与图像错位。

## 敏感性范围如何构建

所有候选共用**同一条基线、同一 ROI、同一排除区域和角度定义**，不人为移动基线来制造变化。候选包括：

1. 二次/三次局部参数曲线，分别使用 0.08、0.12、0.18 的拟合弧长比例。
2. 较弱和较强的轻度高斯去噪（sigma 0.5、1.2），其余保留当前局部拟合设置。
3. 全轮廓圆拟合：只有残差通过检查，且与独立局部左右角均相差不超过 5°、局部左右角差不超过 5°，才纳入综合结果。

候选去重后逐一运行。使用 SciPy 的 soft-L1 鲁棒最小二乘，降低少数离群点的影响；拟合 RMS 超过 1.5 px 的方法不进入综合结果。这些阈值是 v0.1 的可复核筛选规则，尚未对所有实验条件标定。**残差低也不能证明轮廓识别正确。**

每一侧输出：

- `median_deg`：通过检查的候选角度的中位数，作为综合估计；
- `min_deg`、`max_deg`：通过检查的方法之间的最小值和最大值；
- `span_deg`：范围宽度；
- `max_abs_delta_deg`：相对主测量值的最大偏离。

跨度超过默认 3° 时标为敏感；方法失败/被排除或候选不足时要求复核。没有候选通过时，范围和中位数为 null，主测量值仅作诊断，不作为可报告的最终角度。指定局部拟合失败时，会尝试其他局部模型并在 `requested_parameters` 和 `parameters` 中分别记录请求值与实际使用值。

**这是方法敏感性范围，不是置信区间或准确度界限。** 候选共享图像、分割和基线，可能共同受偏；范围较窄也不能排除遮挡或错误基线造成的系统误差。圆拟合会强制左右相同，仅作为适用时的交叉检查。上述标准接口不含 Young–Laplace 拟合、接触点深度遮挡重建或表面自由能计算。

## Python 调用

```python
from pathlib import Path
from contact_angle.core import measure_bytes
from contact_angle.models import MeasurementOptions

options = MeasurementOptions.model_validate_json(
    Path("examples/demo_90.options.json").read_text()
)
output = measure_bytes(Path("examples/demo_90.png").read_bytes(), options)
print(output.result.model_dump_json(indent=2))
Path("annotated.png").write_bytes(output.annotation_png)
```

无法测量时抛出含 `code` 和 `message` 的 `MeasurementError`；参数不合法时 Pydantic 会拒绝请求。

## FastAPI（可选，不影响直接调用）

```bash
contact-angle serve
```

默认仅监听本机 `127.0.0.1:8765`。浏览器打开根路径进入测量工作台。`GET /health` 检查服务，`POST /measure` 保留核心测量接口，`/docs` 提供接口文档。

```bash
curl -X POST http://127.0.0.1:8765/measure \
  -F 'image=@examples/demo_90.png' \
  -F 'options=<examples/demo_90.options.json' \
  -F 'include_annotation=false'
```

`options` 是 multipart 中的 JSON 字符串；设 `include_annotation=true` 可同时返回 PNG 的 Base64。API 不自动保存照片。该服务按本机单用户用途设计，不含公网身份认证和任务队列。

## 验证与来源

```bash
pytest -q
ruff check .
python examples/generate_examples.py
```

`examples/ground_truth.json` 记录合成样例的解析参考角度；带纤维与模糊的图片明确标为合成干扰数据。参考资料：

- [OpenCV 轮廓提取](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [SciPy 鲁棒最小二乘](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
- [FastAPI 文件与表单接口](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/)
- [KRÜSS 接触角拟合方法说明](https://www.kruss-scientific.com/files/kruss-tn314-en.pdf)

未来发布 GitHub 时，可以继续使用本项目结构，添加许可证、自动测试及各系统分发包。

## 本次验证的输出示例

- `sample-results/demo_120/`：120° 合成圆帽，主测量左 119.66°、右 119.71°；9 个候选通过，方法范围分别为 118.94°—120.30°、118.97°—120.36°。
- `sample-results/demo_fuzzy_masked/`：原始圆帽为 90°，叠加模糊、噪声和纤维形干扰后，按已知干扰区域排除部分轮廓。主测量左 90.24°、右 90.21°；7 个方法通过，范围分别为 89.82°—96.51°、89.66°—93.90°，因此仍为 `review`。

这些是合成验证结果，不代表真实照片的误差保证。对应配置见 `examples/demo_fuzzy_masked.options.json`。每个结果目录都含标注图与完整逐方法记录。

## v0.2 纤维表面专用流水线

针对液滴与表面纤维相连的侧视图，新接口会先剔除不连续边缘，再比较左右独立局部圆、按高度拟合的三次曲线，以及全轮廓椭圆模型。使用留出靠近接触区的可见边缘检查外推预测误差，按模型族等权汇总；人工角度不会进入测量接口。

```python
from contact_angle.fibrous import measure_fibrous_bytes
from pathlib import Path
import json

result, annotation_png = measure_fibrous_bytes(Path("drop.bmp").read_bytes())
# 可以传入 baseline=[{"x":50,"y":450},{"x":974,"y":430}]，替代自动基线估计。
Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
Path("annotated.png").write_bytes(annotation_png)
```

返回每侧汇总角、接触点、筛选后方法范围、筛选前全部残差合格候选的范围、模型族数、全部失败/排除原因。只剩一个模型族时会提示复核。灰线为原轮廓、绿线为候选拟合使用的可见点、蓝线为基线、红线为汇总角的接触切线；汇总角由多个模型产生，并不声称有单一曲线与之完全对应。

当前固定协议：基线上方剔除高度 12/20/28 px，侧边保留到液滴高度的 0.3/0.4/0.5，拟合 RMS 和留段预测 RMS 上限 1 px，候选预测误差允许比最佳候选高 0.15 px。全椭圆只在轮廓失配较小、剔除点少于 5% 时作为粗略尖刺检查依据；下部轮廓预测仍独立检查。全部筛选规则依据像素和几何信息，不依据接近人工角度的程度。

这条流水线针对有充分侧轮廓和可辨认表面的 8 位暗滴图片。敏感性固定基线，不能包括共同基线误差。独立合成测试出现过单候选角度偏差 7.59°、范围却为零的反例，因此范围窄不能代替准确度验证。

批量比较可调用 `scripts/compare_fibrous_batch.py --source 图片目录 --before 上一轮records.json --output 新空目录`。验收“通过”定义为左右两侧分别与人工对照绝对差严格小于 5°；也单独输出每图左右均值的验收标记。对照人工值只在测量结束后参与统计。

## 外部 100 图实验路径（2026-11-16）

新增的独立 Young–Laplace 求解器拟合每侧可见轮廓。`general.py` 增加不均匀背景处理、透明薄边检测、局部亚像素定位，以及轮廓完整性与优化收敛检查。它使用基线上方实际可见的侧边，不能恢复被遮挡的真实接触点。

```python
from pathlib import Path
from contact_angle.general import measure_general_bytes

result = measure_general_bytes(
    Path("drop.png").read_bytes(),
    baseline=[{"x": 5, "y": 360}, {"x": 634, "y": 360}],
)
print(result["left"]["angle_deg"], result["right"]["angle_deg"])
```

此路径返回角度、接触点、残差、拟合参数、优化诊断和实际使用的分割模式；不生成标注图，也未实现经校准的敏感性范围。候选按图像拟合残差选择，不能当作独立准确度验证。`baseline=None` 时可以尝试自动估计，但此次 100 图测评使用来源标注或人工几何辅助基线，不能据此宣称全自动精度。

固定样本、分级与参考值后，实验路径按用户最新 **2° / 3° / 5°** 标准，在光滑、轻度、明显噪声三档分别通过 **20/34、28/52、9/14**；三档目标均未达到。早期 **1° / 2.5° / 5°** 标准的结果是 **10/34、26/52、9/14**，此次仅重新计分，未重新拟合。左右参考明确时两侧都必须低于阈值；论文只标一个角度时，单独记录与图像级标量的差。失败留在分母。20 张来源分离的测试图全部属于轻度噪声，仅 2 张通过；后续针对这些图的诊断是探索性开发，不能继续声称其为未查看的验收集。

在项目根目录运行固定数据的完整复算：

```bash
python -m benchmark.runner /absolute/path/cohort/manifest.json \
  --adapter benchmark.adapters:general \
  --before-adapter benchmark.adapters:fibrous \
  --output results/external100
```

100 图的来源、原始输入哈希、基线辅助方式、参考形式、失败原因和逐图评分保存在配套验证包中。文献/仪器读数均未提供逐图不确定度，因此本轮结果衡量与来源读数的一致性。项目未引入 Conan-ML 代码或模型权重；预训练模型和接触角上限约束拟合均只做了独立实验，未用于上述主结果。
