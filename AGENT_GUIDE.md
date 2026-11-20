# Codex / 电脑代理操作接触角工作台

目标是让代理在可复查的界面中定位边界、打点、拟合并保存标注图。所有点位以原始图片像素为单位；参考角度只用于测量完成后的比较，不用于挑选点位或模型。

## 通过界面操作

1. 打开启动器显示的本机 URL，选择或上传目标图片，核对文件名与原图。
2. 自动基线来自液滴两侧附近的固体表面边缘采样。“重新选点”可重设多个表面点；蓝色虚线是它们拟合出的基线。
3. 手动测量选择“轮廓”（C），沿可见边缘打点，第 5 个点会触发轮廓预览。覆盖充分时采用椭圆拟合，短弧采用近圆拟合。边缘可见时，在两侧接触区附近补点。
4. 点击“打点坐标”可进入点表精确输入，展开“导入 / 导出坐标”可应用 JSON。坐标以原图为准。
5. 点击“计算接触角”，检查曲线、交点和角度。结果显示“手动椭圆拟合”或“手动近圆拟合”。下载标注 PNG、JSON、CSV 或 ZIP。
6. 缩放使用加减按钮或 Ctrl／⌘ 加滚轮，范围 25%–800%；普通滚轮不会缩放。H 或空格拖动平移，0 适配。

坐标 JSON 示例（数字只演示格式）：

```json
{
  "baseline_points": [{"x": 80, "y": 300}, {"x": 120, "y": 300}, {"x": 480, "y": 300}, {"x": 520, "y": 300}],
  "contour_points": [{"x": 180, "y": 300}, {"x": 170, "y": 270}, {"x": 200, "y": 220}, {"x": 300, "y": 190}, {"x": 400, "y": 220}, {"x": 430, "y": 270}, {"x": 420, "y": 300}],
  "manual_method": "ellipse",
  "mode": "manual"
}
```

每点需满足 `0 ≤ x < width`、`0 ≤ y < height`。表面点最多 200 个，轮廓点最多 400 个。旧的 `baseline` 两端点与 `left_points` / `right_points` 接口仍可读取；新界面统一使用表面点与轮廓点。

## 通过本机 API 重现点位

API 与界面共用资料库。服务默认只监听 `127.0.0.1`，无需 OpenAI API Key。先通过 `GET /api/info` 核对 `name` 为 `Contact Angle Studio`、版本和资料库目录，避免调用同端口上的其他软件。

| 操作 | API |
|---|---|
| 工作台信息 | `GET /api/info` |
| 图片列表 | `GET /api/images` |
| 上传图片 | `POST /api/images`，multipart字段 `image` |
| 原图预览 | `GET /api/images/{image_id}/preview.png` |
| 自动基线建议 | `GET /api/images/{image_id}/baseline` |
| 预览手动表面点拟合 | `POST /api/images/{image_id}/baseline`，提交草稿格式的参数和 `baseline_source: "manual"` |
| 读取 / 保存当前点位草稿 | `GET / PUT /api/images/{image_id}/draft` |
| 提交测量 | `POST /api/jobs` |
| 查询测量状态 | `GET /api/jobs/{job_id}` |
| 该图片的结果历史 | `GET /api/images/{image_id}/results` |
| 标注 PNG | `GET /api/results/{job_id}/annotation.png` |
| JSON / CSV / ZIP | `GET /api/results/{job_id}/result.json`、`results.csv`、`bundle.zip` |

手动测量请求使用 `{"image_id":"图片ID","options":{...}}`，其中 `options` 使用上面的 `baseline_points`、`contour_points`、`manual_method: "ellipse"`、`mode: "manual"`。设置 `sensitivity: true` 比较点位与表面位置变化下的角度。自动模式使用 `mode: "auto"`；轮廓点不会参与自动轮廓识别。

接口返回202后，按任务ID查询状态。`queued` 和 `running` 尚未完成；`done` 的 `result` 包含左右角、基线、几何曲线、敏感性和导出链接；`error` 的原因应保留并告知用户。不要在更改图片或点位后继续显示旧结果为当前结果。

自动测量使用 `mode: "auto"`；`algorithm` 可为 `general`、`fibrous`、`local`。`baseline: null` 表示需要自动基线；明确提供基线时同时设置对应的 `baseline_source`。所有方法选择和适用性仍应结合原图检查。

倒置液滴直接使用原始图像。工作台的 `orientation` 可取 `auto`（默认）、`pendant`（液滴在下）或 `sessile`（液滴在上）。悬滴结果中的 `surface_curve` 描述表面，`left.surface_tangent` 和 `right.surface_tangent` 是指向液相内侧的局部表面切线。用于兼容的两点 `baseline` 不能代替曲面求角。

## 交付用户的测量记录

保留文件名、原图SHA256、原图尺寸、基线、点位、所用方法、左右角、敏感性范围和复核提示。保存有接触角标注的 PNG，方便用户直接核对。多图汇总时保留失败图片的条目；报告均值时说明它与左右独立角的区别。
