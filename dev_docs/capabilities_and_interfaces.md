# 现有能力与接口梳理

> 用途:为后续方案分析(尤其是语义选区布尔操作)提供一份事实基线。
> 覆盖范围:server(`server/nninteractive_slicer_server/main.py`)与
> client(`slicer_plugin/SlicerNNInteractive/SlicerNNInteractive.py`)。
> 所有行号基于评审时的代码,改动后需复核。

---

## 一、Server 端(FastAPI)

进程内只有一个全局 `PROMPT_MANAGER`(`main.py:269`),持有**单一图像 + 单一推理会话**,
是**有状态、单会话**的。多个客户端并发会互相覆盖。

### 1.1 HTTP 接口

| 方法/路径 | 请求体 | 作用 | 返回 |
|-----------|--------|------|------|
| `POST /upload_image` | multipart 文件,未压缩 `.npy`(3D 数组) | `set_image()`:载入图像、重置交互 | `{"status":"ok"}` |
| `POST /upload_segment` | multipart 文件,gzip 压缩的 `.npy` mask | `set_segment()`:设置/重置初始分割 | `{"status":"ok"}` |
| `POST /add_point_interaction` | JSON `PointParams` | 点提示推理 | gzip 压缩的位打包二值 mask |
| `POST /add_bbox_interaction` | JSON `BBoxParams` | 包围盒提示推理 | 同上 |
| `POST /add_lasso_interaction` | multipart:`file`(gzip `.npy` 3D mask)+ `positive_click`(form 字符串) | lasso 提示推理 | 同上 |
| `POST /add_scribble_interaction` | multipart:同 lasso | scribble 提示推理 | 同上 |

请求模型:
- `PointParams`:`voxel_coord: list[int]`、`positive_click: bool`(`main.py:316`)
- `BBoxParams`:`outer_point_one: list[int]`、`outer_point_two: list[int]`、`positive_click: bool`(`main.py:348`)

错误返回:图像未上传时返回 `{"status":"error","message":"No image uploaded"}`
(`get_error_if_img_not_set`,`main.py:126`)。客户端据此自动重传重试。

### 1.2 `PromptManager` 方法(`main.py:137`)

| 方法 | 作用 |
|------|------|
| `download_weights()` | 从 HuggingFace 下载 `nnInteractive_v1.0` 权重到 `.nninteractive_weights/` |
| `make_session()` | 创建 `nnInteractiveInferenceSession`,设备由 `get_inference_device()` 决定 |
| `set_image(arr)` | `reset_interactions()` → 存图(补成 `(1,x,y,z)`)→ 新建全 0 `target_tensor` |
| `set_segment(mask)` | mask 全 0:`reset_interactions()` + 新 target buffer;mask 非空:`add_initial_seg_interaction(mask)` |
| `add_point_interaction(coords, include)` | 调 `session.add_point_interaction`,返回 `target_tensor` 的 numpy 副本 |
| `add_bbox_interaction(p1, p2, include)` | 把两角点归一成 `[[min,max],...]` 后调 session,返回 mask |
| `add_lasso_interaction(mask, include)` | 调 `session.add_lasso_interaction`,返回 mask |
| `add_scribble_interaction(mask, include)` | 调 `session.add_scribble_interaction`,返回 mask |

> **注意**:`set_segment` 在 mask 非空时**未先 `reset_interactions()`** 就 `add_initial_seg_interaction`。
> 布尔编辑后回传 mask 时,该行为是否符合预期需实测确认(见评审文档待确认事项)。

### 1.3 数据格式辅助函数

- `segmentation_binary(seg, compress)`(`main.py:91`):bool 数组 → `np.packbits` 位打包 →(可选)gzip。
- `unpack_binary_segmentation(data, vol_shape)`(`main.py:77`):反向操作,客户端同名方法对应。
- 所有推理结果以 `Response(content=..., media_type="application/octet-stream",
  headers={"Content-Encoding":"gzip"})` 返回。
- `get_inference_device()`(`main.py:37`):`NNI_DEVICE` 环境变量 > `cuda:0` > CPU(带告警)。

---

## 二、Client 端(`SlicerNNInteractiveWidget`)

整个客户端是一个大类。下面按职责分组列出关键方法与状态。

### 2.1 核心状态

| 字段 | 含义 |
|------|------|
| `self.server` | server URL(去尾 `/`),持久化于 QSettings `SlicerNNInteractive/server` |
| `self.previous_states` | `dict`,键 `"image_data"` / `"segment_data"`,用于 diff 判断是否需要重传 |
| `self.prompt_types` | point / bbox / lasso 三类提示的配置(节点类、按钮、回调等) |
| `self.segment_editor_node` | 与核心 Segment Editor 模块共享的 `vtkMRMLSegmentEditorNode`(singleton) |
| `self.ui.editor_widget` | 嵌入的 `qMRMLSegmentEditorWidget`,**所有分割都作用于它当前选中的 segment** |
| `self.scribble_segment_node` | 隐藏节点 `"ScribbleSegmentNode (do not touch)"`,scribble 专用脚手架 |
| `self._prev_scribble_mask` | 上一次 scribble 笔画 mask,用于发送笔画差分 |

### 2.2 同步机制:`@ensure_synched`(`SlicerNNInteractive.py:40`)

装饰所有 prompt 方法(`point_prompt`、`bbox_prompt`、`lasso_or_scribble_prompt`)。
每次 prompt 前:

1. `image_changed()` 为真 → `upload_image_to_server()`。
2. `selected_segment_changed()` 为真 → `remove_all_but_last_prompt()` + `upload_segment_to_server()`。
3. 两者都成功才执行真正的 prompt 函数。

判定靠 `np.array_equal` 比较当前数据与 `previous_states`。
**含义:用户用任何方式改了当前 segment,下一次 prompt 会自动把新 mask 同步到 server。**
此外 `request_to_server` 在收到 "No image uploaded" 时会自动重传图像+分割并重试。

### 2.3 方法分组

**Setup / 生命周期**
`setup` `init_ui_functionality` `setup_shortcuts` `remove_shortcut_items`
`install_dependencies` `check_dependency_installed` `pip_install_wrapper`
`run_with_progress_bar` `cleanup` `__del__`

**提示与 Markup 管理**
`setup_prompts` `setup_scribble_prompt` `is_ui_dark_or_light_mode`
`remove_prompt_nodes` `on_interaction_node_modified` `remove_all_but_last_prompt`
`on_place_button_clicked` `display_node_markup_point/bbox/lasso`

**提示事件与上传**
`on_point_placed` → `point_prompt`(`@ensure_synched`)
`on_bbox_placed` → `bbox_prompt`(`@ensure_synched`)
`on_lasso_placed` `on_lasso_cancel_clicked` `submit_lasso_if_present`
`on_scribble_clicked` `on_scribble_finished` → `lasso_or_scribble_prompt`(`@ensure_synched`)

**分割操作(布尔方案最相关)**

| 方法 | 行号 | 作用 |
|------|------|------|
| `get_segment_data()` | `:1068` | 取当前选中 segment 的二值 numpy mask(bool,`(z,y,x)`) |
| `show_segmentation(mask)` | `:979` | 把 numpy mask 写回当前 segment;含 `saveStateForUndo()`、更新 `previous_states["segment_data"]`、按需重建 3D 表面 |
| `make_new_segment()` | `:920` | 新建并选中 `Segment_N` |
| `clear_current_segment()` | `:958` | 清空当前 segment 并上传 server |
| `get_segmentation_node()` | `:1020` | 取/建分割节点(**刻意排除** scribble 隐藏节点) |
| `get_selected_segmentation_node_and_segment_id()` | `:1049` | 取当前节点+segment id,无则新建 |
| `get_current_segment_id()` | `:1062` | 当前选中 segment id |
| `selected_segment_changed()` | `:1083` | 当前 segment 是否相对 `previous_states` 变化 |

**Server 通信**
`update_server` `test_server_connection` `request_to_server`
`upload_image_to_server` `upload_segment_to_server`

**工具/转换**
`get_image_data` `get_volume_node` `image_changed` `mask_to_np_upload_file`
`unpack_binary_segmentation` `ras_to_xyz` `xyz_from_caller` `lasso_points_to_mask`

**提示极性切换**
`is_positive`(property) `on_prompt_type_positive_clicked`
`on_prompt_type_negative_clicked` `toggle_prompt_type`

### 2.4 三类提示的 Slicer 节点

| 提示 | Markup 节点类 | 触发方式 |
|------|---------------|----------|
| point | `vtkMRMLMarkupsFiducialNode` | 放点 → `PointPositionDefinedEvent` |
| bbox | `vtkMRMLMarkupsROINode` | 放两个角点 |
| lasso | `vtkMRMLMarkupsClosedCurveNode` | 闭合曲线 → 栅格化为单层 mask |
| scribble | 隐藏 `qMRMLSegmentEditorWidget` + Paint 效果 | 发送相邻笔画的差分 mask |

### 2.5 键盘快捷键(`setup_shortcuts`,`:225`)

`o` 点 / `b` 包围盒 / `l` lasso / `s` scribble / `e` 新建 segment /
`r` 清空 segment / `Shift+L` 提交 lasso / `t` 正负极性切换。

---

## 三、坐标与数据格式约定

- numpy volume / mask 一律 `(z, y, x)` 排列。
- 与模型交换的体素坐标是 `(x, y, z)`;客户端发送 point/bbox 前用 `xyz[::-1]` 反转。
- `ras_to_xyz()` 返回 `(x, y, z)` 整数;构造 numpy mask 时需注意换序。
- 上传图像:未压缩 `.npy`;上传 mask 与 lasso/scribble:gzip 压缩 `.npy`。
- server 返回:gzip + 位打包,客户端 `unpack_binary_segmentation` 还原成 volume 形状。

---

## 四、可复用的 Slicer 原生能力(布尔方案相关)

客户端嵌入的 `self.ui.editor_widget` 是完整的 `qMRMLSegmentEditorWidget`,
以下原生 Segment Editor 效果**当前已可用**,无需自研:

| 原生效果 | 能力 | 与布尔方案的关系 |
|----------|------|------------------|
| **Logical operators** | segment 间 `Copy / Add(并) / Subtract(差) / Intersect(交) / Invert / Clear / Fill` | 直接覆盖"segment-to-segment 布尔操作" |
| **Scissors** | 在 2D/3D 视图用矩形或自由形状切割/保留 segment | 直接覆盖"ROI box 切割",且免去手动 RAS→IJK 栅格化 |
| **Paint / Draw / Erase** | 手动绘制 mask | 可作为"临时操作选区"的输入来源 |
| **Threshold / Islands / Smoothing** | 阈值、连通域、平滑 | 后续精修可用 |

相关 Slicer API:
- `slicer.util.arrayFromSegmentBinaryLabelmap(node, segId, refVolume)` — segment → numpy。
- `slicer.util.updateSegmentBinaryLabelmapFromArray(...)` — numpy → segment(`show_segmentation` 已用)。
- `vtkMRMLMarkupsROINode` — 3D 包围盒节点(bbox 提示已用)。

---

## 五、扩展点(新功能可挂接的位置)

| 需求 | 可挂接位置 |
|------|------------|
| 写回布尔结果 | 复用 `show_segmentation(mask)`,已含 undo 与状态更新 |
| 取 operand segment 数据 | `arrayFromSegmentBinaryLabelmap(...)` 或 `get_segment_data()` |
| 立即同步到 server | 直接调 `upload_segment_to_server()`(不经 `@ensure_synched`) |
| 新增 UI 区块 | 仿照 `init_ui_functionality` 连接控件;UI 定义在 `Resources/UI/SlicerNNInteractive.ui` |
| 新增临时编辑节点 | 仿照 `setup_scribble_prompt` 的隐藏节点模式 |
| 新增快捷键 | 在 `setup_shortcuts` 的 `shortcuts` 字典加项 |

---

## 六、给后续方案分析的关键事实

1. server 是单会话、有状态的;布尔编辑无需新增 server API。
2. `@ensure_synched` 已能在下一次 prompt 时**惰性**同步本地 segment 改动;
   缺的只是一个"立即同步"入口。
3. segment 间布尔运算、ROI 切割在 Slicer 原生已有(Logical operators / Scissors),
   重点应放在 UI 引导与 server 同步,而非重写算法。
4. 所有分割都作用于 `editor_widget` 当前选中的 segment;
   `"ScribbleSegmentNode (do not touch)"` 是内部节点,不可当作普通结构。
5. 几何一致性以当前 source volume 为基准;跨 volume/geometry 的 mask 直接布尔会出错。
